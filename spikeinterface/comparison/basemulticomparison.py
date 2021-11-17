from typing import OrderedDict
import numpy as np
import networkx as nx
from copy import deepcopy
from .basecomparison import BaseComparison


class BaseMultiComparison(BaseComparison):
    """
    Compares multiple comparison objects using a graph approach.

    - Pair-wise comparisons are made
    - An agreement graph is built based on the agreement score

    Parameters
    ----------
    object_list: list
        List of objects to be compared
    name_list: list
        List of object names
    """

    def __init__(self, object_list, name_list=None):
        self._object_list = object_list
        self.name_list = name_list
        self.graph = None
        self.subgraphs = None
        self.clean_graph = None

    def _compute_all(self):
        self._do_comparison()
        self._do_graph()
        self._clean_graph()
        self._do_agreement()

    def _compare_ij(self, i, j):
        raise NotImplementedError

    def _populate_nodes(self):
        raise NotImplementedError

    @property
    def units(self):
        return deepcopy(self._new_units)

    def compute_subgraphs(self):
        """
        Computes subgraphs of connected components.

        Returns
        -------
        sg_object_names: list
            List of sorter names for each node in the connected component subgraph
        sg_units: list
            List of unit ids for each node in the connected component subgraph
        """
        if self.clean_graph is not None:
            g = self.clean_graph
        else:
            g = self.graph
        subgraphs = (g.subgraph(c).copy() for c in nx.connected_components(g))
        sg_object_names = []
        sg_units = []
        for sg in subgraphs:
            object_names = []
            unit_names = []
            for node in sg.nodes:
                object_names.append(node[0])
                unit_names.append(node[1])
            sg_object_names.append(object_names)
            sg_units.append(unit_names)
        return sg_object_names, sg_units

    def _do_comparison(self, ):
        # do pairwise matching
        if self._verbose:
            print('Multicomaprison step 1: pairwise comparison')

        self.comparisons = []
        for i in range(len(self._object_list)):
            for j in range(i + 1, len(self._object_list)):
                if self._verbose:
                    if self.name_list is not None:
                        name_i = self.name_list[i]
                        name_j = self.name_list[j]
                    else:
                        name_i = "object i"
                        name_j = "object j"
                    print(f"  Comparing: {name_i} and {name_j}")
                comp = self._compare_ij(i, j)
                self.comparisons.append(comp)

    def _do_graph(self):
        if self._verbose:
            print('Multicomparison step 2: make graph')

        self.graph = nx.Graph()
        # nodes
        self._populate_nodes()

        # edges
        for comp in self.comparisons:
            for u1 in comp.hungarian_match_12.index.values:
                u2 = comp.hungarian_match_12[u1]
                if u2 != -1:
                    node1 = comp.name_list[0], u1
                    node2 = comp.name_list[1], u2
                    score = comp.agreement_scores.loc[u1, u2]
                    self.graph.add_edge(node1, node2, weight=score)

        # the graph is symmetrical
        self.graph = self.graph.to_undirected()

    def _clean_graph(self):
        if self._verbose:
            print('Multicomaprison step 3: clean graph')
        clean_graph = self.graph.copy()
        subgraphs = (clean_graph.subgraph(c).copy()
                     for c in nx.connected_components(clean_graph))
        removed_nodes = 0
        for sg in subgraphs:
            object_names = []
            for node in sg.nodes:
                object_names.append(node[0])
            sorters, counts = np.unique(object_names, return_counts=True)

            if np.any(counts > 1):
                for sorter in sorters[counts > 1]:
                    nodes_duplicate = [n for n in sg.nodes if sorter in n]
                    # get edges
                    edges_duplicates = []
                    weights_duplicates = []
                    for n in nodes_duplicate:
                        edges = sg.edges(n, data=True)
                        for e in edges:
                            edges_duplicates.append(e)
                            weights_duplicates.append(e[2]['weight'])
                    # remove edges
                    edges_to_remove = len(nodes_duplicate) - 1
                    remove_idxs = np.argsort(weights_duplicates)[
                        :edges_to_remove]
                    for idx in remove_idxs:
                        if self._verbose:
                            print('Removed edge', edges_duplicates[idx])
                        clean_graph.remove_edge(
                            edges_duplicates[idx][0], edges_duplicates[idx][1])
                        sg.remove_edge(
                            edges_duplicates[idx][0], edges_duplicates[idx][1])
                        if edges_duplicates[idx][0] in nodes_duplicate:
                            sg.remove_node(edges_duplicates[idx][0])
                        else:
                            sg.remove_node(edges_duplicates[idx][1])
                        removed_nodes += 1
        if self._verbose:
            print(f'Removed {removed_nodes} duplicate nodes')
        self.clean_graph = clean_graph

    def _do_agreement(self):
        # extract agreement from graph
        if self._verbose:
            print('Multicomparison step 4: extract agreement from graph')

        self._new_units = {}

        # Save new units
        self.subgraphs = (self.clean_graph.subgraph(c).copy()
                          for c in nx.connected_components(self.clean_graph))
        for new_unit, sg in enumerate(self.subgraphs):
            edges = list(sg.edges(data=True))
            if len(edges) > 0:
                avg_agr = np.mean([d['weight'] for u, v, d in edges])
            else:
                avg_agr = 0
            object_unit_ids = {}
            for node in sg.nodes:
                object_name, unit_name = node
                object_unit_ids[object_name] = unit_name
            # sort dict based on name list
            sorted_object_unit_ids = OrderedDict()
            for name in self.name_list:
                if name in object_unit_ids:
                    sorted_object_unit_ids[name] = object_unit_ids[name]
            self._new_units[new_unit] = {'avg_agreement': avg_agr, 'unit_ids': sorted_object_unit_ids,
                                         'agreement_number': len(sg.nodes)}

