# NERO

Network Embedding based on Relation Order histograms

## Installation

While in project main directory (containing the `Dockerfile` file), run the following command.
```bash
docker build -t nero:latest .
```
Then use the prepared image as a remote Python intepreter. Remember to bind the appropriate container volumes (code and storage locations).
```bash
docker run -it --rm --name nero_interpreter -v {project directory path}:/projects/NERO -v {desired storage and output path}:/nero --entrypoint python3 nero /projects/NERO/main.py
```

## Usage

The graph embedding tool can be set up using an object-oriented API.
```python
from nero import constants
from nero.embedding import compressors, digitisers, embedders
from nero.tools import distributions


embedder = embedders.NeroEmbedder(
        bin_generators=[
            digitisers.LabelBinGenerator(
                relevant_property=digitisers.RelevantProperty(
                    name="vertex_label_0", 
                    origin=digitisers.VertexOrigin.NODE,
                ),
            ),
            digitisers.EqualProbabilityBinGenerator(
                relevant_property=digitisers.RelevantProperty(
                    name="edge_attribute_0", 
                    origin=digitisers.VertexOrigin.EDGE,
                ),
                bins_no=20,
                insight_bins_generator=digitisers.InsightBinGenerator(
                    calculate_bins=distributions.equal_size_bins,
                    bins_no=constants.INSIGHT_BINS_NO,
                    bin_type='equal',
                ),
            ),
        ],
        relation_order_compressor=compressors.VGGInspiredROC(
            bins_no=20,
        ),
        jobs_no=8,
    )
```
It is recommended to use the aforementioned embedder as a part of a scikit-learn processing pipeline.
```python
import sklearn.pipeline as skl_pipeline
import sklearn.preprocessing as skl_preprocessing
import sklearn.ensemble as skl_ensemble

from nero.pipeline.transform import normalise, resize


pipeline = skl_pipeline.Pipeline([
        ('embed', embedder),
        ('resize_to_smallest_common_shape', resize.ToSmallestCommonShape()),
        ('normalise', normalise.PercentOfMaxPerRelationOrder()),
        ('flatten', resize.FlattenUpperTriangles()),
        ('scale', skl_preprocessing.StandardScaler()),
        ('classify', skl_ensemble.ExtraTreesClassifier(
            n_estimators=1000,
            n_jobs=8,
            verbose=False
        )),
    ])
```
For more detailed usage information check the demonstration example enclosed in the source code and the accompanying conference paper.

## BibTeX
```bibtex
@inproceedings{lazarz2021relation,
  title={Relation Order Histograms as a Network Embedding Tool},
  author={{\L}azarz, Rados{\l}aw and Idzik, Micha{\l}},
  booktitle={International Conference on Computational Science},
  pages={224--237},
  year={2021},
  organization={Springer}
}
```

 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/relation-order-histograms-as-a-network/graph-classification-on-dd)](https://paperswithcode.com/sota/graph-classification-on-dd?p=relation-order-histograms-as-a-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/relation-order-histograms-as-a-network/graph-classification-on-proteins)](https://paperswithcode.com/sota/graph-classification-on-proteins?p=relation-order-histograms-as-a-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/relation-order-histograms-as-a-network/graph-classification-on-nci1)](https://paperswithcode.com/sota/graph-classification-on-nci1?p=relation-order-histograms-as-a-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/relation-order-histograms-as-a-network/graph-classification-on-mutag)](https://paperswithcode.com/sota/graph-classification-on-mutag?p=relation-order-histograms-as-a-network)
