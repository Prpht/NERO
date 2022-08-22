from dataclasses import dataclass
import pathlib
from typing import Any, Dict, Tuple

import graph_tool as gt
import networkx as nx

import nero.constants as constants
import nero.converters.nx2gt as nx2gt


@dataclass
class ClassificationSample:
    category_attr: str
    numerical_category: int
    verbose_category: str
    domain_attrs: Dict[str, Any]


@dataclass
class MaterialisedClassificationSample(ClassificationSample):
    graph: gt.Graph


@dataclass
class PersistedClassificationSample(ClassificationSample):
    graph_file_name: str
    dataset_type: str

    def materialise(self) -> MaterialisedClassificationSample:
        root_path = pathlib.Path(constants.PICKLES_DIR)
        graph_file_path = str(root_path / self.dataset_type / self.category_attr / self.graph_file_name)
        graph = gt.load_graph(graph_file_path)
        return MaterialisedClassificationSample(
            self.category_attr, self.numerical_category, self.verbose_category, self.domain_attrs, graph
        )


def create_persisted_sample(
        graph: nx.Graph,
        dataset_name: str,
        i: int,
        supervision_target: str,
) -> PersistedClassificationSample:
    verbose_category = f"category_{graph.graph[supervision_target]}"
    numerical_category = graph.graph[supervision_target]
    graph_file_name, domain_attrs = convert_and_persist_graph(graph, dataset_name, i, supervision_target)
    return PersistedClassificationSample(
        supervision_target,
        numerical_category,
        verbose_category,
        domain_attrs,
        graph_file_name,
        str(pathlib.Path(constants.PICKLES_DIR) / dataset_name),
    )


def convert_and_persist_graph(
        nx_graph: nx.Graph,
        dataset_name: str,
        i: int,
        supervision_target: str
) -> Tuple[str, Dict[str, Any]]:
    gt_graph = nx2gt.nx2gt(nx_graph)
    graph_file_name = f"{dataset_name}_{i}.gt"
    store_path = pathlib.Path(constants.PICKLES_DIR) / dataset_name / supervision_target
    try:
        store_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    gt_graph.save(str(store_path / graph_file_name))
    domain_attrs = extract_domain_attributes(gt_graph)
    return graph_file_name, domain_attrs


def extract_domain_attributes(gt_graph: gt.Graph) -> Dict[str, Any]:
    domain_attrs = {}
    for graph_property in gt_graph.graph_properties:
        domain_attrs[str(graph_property)] = gt_graph.graph_properties[graph_property]
    return domain_attrs
