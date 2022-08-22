from typing import List, Optional, Tuple

from sklearn import pipeline as skl_pipeline, preprocessing as skl_preprocessing, ensemble as skl_ensemble

from nero import constants
from nero.converters import tudataset
from nero.embedding import compressors, digitisers, embedders
from nero.pipeline.transform import normalise, resize, utils
from nero.tools import distributions

TREES_NO = 1000
JOBS_NO = 8


class UnknownTagException(Exception):
    pass


def create_labels_bin_generator(
        origin: digitisers.VertexOrigin,
        property_no: int,
) -> digitisers.LabelBinGenerator:
    return digitisers.LabelBinGenerator(
        relevant_property=digitisers.RelevantProperty(f'{origin}_label_{property_no}', origin),
        disable_tqdm=True,
    )


def create_equal_probability_bin_generator(
        origin: digitisers.VertexOrigin,
        property_no: int,
        bins_no: int,
) -> digitisers.EqualProbabilityBinGenerator:
    return digitisers.EqualProbabilityBinGenerator(
        relevant_property=digitisers.RelevantProperty(f'{origin}_attribute_{property_no}', origin),
        bins_no=bins_no,
        insight_bins_generator=digitisers.InsightBinGenerator(
            calculate_bins=distributions.equal_size_bins,
            bins_no=constants.INSIGHT_BINS_NO,
            bin_type='equal',
        ),
        disable_tqdm=True,
    )


def create_equal_size_bin_generator(
        origin: digitisers.VertexOrigin,
        property_no: int,
        bins_no: int,
) -> digitisers.EqualSizeBinGenerator:
    return digitisers.EqualSizeBinGenerator(
        relevant_property=digitisers.RelevantProperty(f'{origin}_attribute_{property_no}', origin),
        bins_no=bins_no,
        disable_tqdm=True,
    )


def create_bin_generators(
        description: tudataset.TUDatasetDescription,
        tag: str,
        parameter: Optional[int],
) -> List[digitisers.BinGenerator]:
    parameter = parameter if parameter is not None else 10
    result: List[digitisers.BinGenerator] = []

    for i in range(description.node_labels):
        result.append(create_labels_bin_generator(digitisers.VertexOrigin.NODE, i))
    for i in range(description.edge_labels):
        result.append(create_labels_bin_generator(digitisers.VertexOrigin.EDGE, i))

    if tag == '0':
        create_bin_generator = create_equal_size_bin_generator
    elif tag == 'A':
        create_bin_generator = create_equal_probability_bin_generator
    else:
        raise UnknownTagException(f"Unknown generator tag: {tag}!")

    for i in range(description.node_attributes):
        result.append(create_bin_generator(digitisers.VertexOrigin.NODE, i, parameter))
    for i in range(description.edge_attributes):
        result.append(create_bin_generator(digitisers.VertexOrigin.EDGE, i, parameter))
    return result


def create_compressor(tag: str, parameter: Optional[int]) -> compressors.RelationOrderCompressor:
    if tag == '0':
        return compressors.NoCompressionROC()
    else:
        parameter = parameter if parameter is not None else 10
        if tag == 'V':
            return compressors.VGGInspiredROC(parameter)
        elif tag == 'F':
            return compressors.FibonacciROC(parameter)
        else:
            raise UnknownTagException(f"Unknown compressor tag: {tag}!")


def create_normaliser(tag: str) -> utils.ReplacingTransformer:
    if tag == '0':
        return normalise.DoNothing(disable_tqdm=True)
    elif tag == 'R':
        return normalise.PercentOfMaxPerRelationOrder(disable_tqdm=True)
    else:
        raise UnknownTagException(f"Unknown normaliser tag: {tag}!")


def create_pipeline(
        description: tudataset.TUDatasetDescription,
        tag: str,
        subvariant_parameters: Tuple[Optional[int], Optional[int], Optional[int]] = (None, None, None)
) -> skl_pipeline.Pipeline:
    digitiser_tag, compressor_tag, normaliser_tag = (tag[0], tag[1], tag[2])
    digitiser_parameter, compressor_parameter, _ = subvariant_parameters

    embedder = embedders.NeroEmbedder(
        bin_generators=create_bin_generators(description, digitiser_tag, digitiser_parameter),
        relation_order_compressor=create_compressor(compressor_tag, compressor_parameter),
        jobs_no=JOBS_NO,
        disable_tqdm=True,
    )
    classifier = skl_ensemble.ExtraTreesClassifier(
        n_estimators=TREES_NO,
        n_jobs=JOBS_NO,
        verbose=False
    )
    return skl_pipeline.Pipeline([
        ('embed', embedder),
        ('resize_to_smallest_common_shape', resize.ToSmallestCommonShape(disable_tqdm=True)),
        ('normalise', create_normaliser(normaliser_tag)),
        ('flatten', resize.FlattenUpperTriangles(disable_tqdm=True)),
        ('scale', skl_preprocessing.StandardScaler()),
        ('classify', classifier),
    ])
