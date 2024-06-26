from __future__ import annotations

from copy import deepcopy
from typing import OrderedDict
import numpy as np

from .comparisontools import make_possible_match, make_best_match, make_hungarian_match


class BaseComparison:
    """
    Base class for all comparisons (SpikeTrain and Template)
    """

    def __init__(self, object_list, name_list, match_score=0.5, chance_score=0.1, verbose=False):
        self.object_list = object_list
        self.name_list = name_list
        self._verbose = verbose
        self.match_score = match_score
        self.chance_score = chance_score


class BaseMultiComparison(BaseComparison):
    """
    Base class for graph-based multi comparison classes.

    It handles graph operations, comparisons, and agreements.
    """

    def __init__(self, object_list, name_list, match_score=0.5, chance_score=0.1, verbose=False):
        import networkx as nx

        BaseComparison.__init__(
            self,
            object_list=object_list,
            name_list=name_list,
            match_score=match_score,
            chance_score=chance_score,
            verbose=verbose,
        )
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

        import networkx as nx

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

    def _do_comparison(
        self,
    ):
        # do pairwise matching
        if self._verbose:
            print("Multicomparison step 1: pairwise comparison")

        self.comparisons = {}
        for i in range(len(self.object_list)):
            for j in range(i + 1, len(self.object_list)):
                if self.name_list is not None:
                    name_i = self.name_list[i]
                    name_j = self.name_list[j]
                else:
                    name_i = "object i"
                    name_j = "object j"
                if self._verbose:
                    print(f"  Comparing: {name_i} and {name_j}")
                comp = self._compare_ij(i, j)
                self.comparisons[(name_i, name_j)] = comp

    def _do_graph(self):
        if self._verbose:
            print("Multicomparison step 2: make graph")

        import networkx as nx

        self.graph = nx.Graph()
        # nodes
        self._populate_nodes()

        # edges
        for comp_name, comp in self.comparisons.items():
            for u1 in comp.hungarian_match_12.index.values:
                u2 = comp.hungarian_match_12[u1]
                if u2 != -1:
                    name_1, name_2 = comp_name
                    node1 = name_1, u1
                    node2 = name_2, u2
                    score = comp.agreement_scores.loc[u1, u2]
                    self.graph.add_edge(node1, node2, weight=score)

        # the graph is symmetrical
        self.graph = self.graph.to_undirected()

    def _clean_graph(self):
        if self._verbose:
            print("Multicomparison step 3: clean graph")
        clean_graph = self.graph.copy()
        import networkx as nx

        subgraphs = (clean_graph.subgraph(c).copy() for c in nx.connected_components(clean_graph))
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
                            weights_duplicates.append(e[2]["weight"])

                    # remove extra edges
                    n_edges_to_remove = len(nodes_duplicate) - 1
                    remove_idxs = np.argsort(weights_duplicates)[:n_edges_to_remove]
                    edges_to_remove = np.array(edges_duplicates, dtype=object)[remove_idxs]

                    for edge_to_remove in edges_to_remove:
                        clean_graph.remove_edge(edge_to_remove[0], edge_to_remove[1])
                        sg.remove_edge(edge_to_remove[0], edge_to_remove[1])
                        if self._verbose:
                            print(f"Removed edge: {edge_to_remove}")

                    # remove extra nodes (as a second step to not affect edge removal)
                    for edge_to_remove in edges_to_remove:
                        if edge_to_remove[0] in nodes_duplicate:
                            node_to_remove = edge_to_remove[0]
                        else:
                            node_to_removed = edge_to_remove[1]
                        if node_to_remove in sg.nodes:
                            sg.remove_node(node_to_remove)
                            print(f"Removed node: {node_to_remove}")
                            removed_nodes += 1

        if self._verbose:
            print(f"Removed {removed_nodes} duplicate nodes")
        self.clean_graph = clean_graph

    def _do_agreement(self):
        # extract agreement from graph
        if self._verbose:
            print("Multicomparison step 4: extract agreement from graph")

        self._new_units = {}

        # save new units
        import networkx as nx

        self.subgraphs = [self.clean_graph.subgraph(c).copy() for c in nx.connected_components(self.clean_graph)]
        for new_unit, sg in enumerate(self.subgraphs):
            edges = list(sg.edges(data=True))
            if len(edges) > 0:
                avg_agr = np.mean([d["weight"] for u, v, d in edges])
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
            self._new_units[new_unit] = {
                "avg_agreement": avg_agr,
                "unit_ids": sorted_object_unit_ids,
                "agreement_number": len(sg.nodes),
            }


class BasePairComparison(BaseComparison):
    """
    Base class for pair comparisons.

    It handles the matching procedurs.

    Agreement scores must be computed in inherited classes by overriding the
    "_do_agreement(self)" function
    """

    def __init__(self, object1, object2, name1, name2, match_score=0.5, chance_score=0.1, verbose=False):
        BaseComparison.__init__(
            self,
            object_list=[object1, object2],
            name_list=[name1, name2],
            match_score=match_score,
            chance_score=chance_score,
            verbose=verbose,
        )
        self.possible_match_12, self.possible_match_21 = None, None
        self.best_match_12, self.best_match_21 = None, None
        self.hungarian_match_12, self.hungarian_match_21 = None, None
        self.agreement_scores = None

    def _do_agreement(self):
        # populate self.agreement_scores
        raise NotImplementedError

    def _do_matching(self):
        if self._verbose:
            print("Matching...")

        self.possible_match_12, self.possible_match_21 = make_possible_match(self.agreement_scores, self.chance_score)
        self.best_match_12, self.best_match_21 = make_best_match(self.agreement_scores, self.chance_score)
        self.hungarian_match_12, self.hungarian_match_21 = make_hungarian_match(self.agreement_scores, self.match_score)

    def get_ordered_agreement_scores(self):
        assert self.agreement_scores is not None, "'agreement_scores' have not been computed!"
        # order rows
        order0 = self.agreement_scores.max(axis=1).argsort()
        scores = self.agreement_scores.iloc[order0.values[::-1], :]

        # order columns
        indexes = np.arange(scores.shape[1])
        order1 = []
        for r in range(scores.shape[0]):
            possible = indexes[~np.isin(indexes, order1)]
            if possible.size > 0:
                ind = np.argmax(scores.iloc[r, possible].values)
                order1.append(possible[ind])
        remain = indexes[~np.isin(indexes, order1)]
        order1.extend(remain)
        scores = scores.iloc[:, order1]

        return scores


class MixinSpikeTrainComparison:
    """
    Mixin for spike train comparisons to define:
       * delta_time / delta_frames
       * sampling frequency
       * n_jobs
    """

    def __init__(self, delta_time=0.4, n_jobs=-1):
        self.delta_time = delta_time
        self.n_jobs = n_jobs
        self.sampling_frequency = None
        self.delta_frames = None

    def set_frames_and_frequency(self, sorting_list):
        sorting0 = sorting_list[0]
        # check num segments
        if not np.all(sorting.get_num_segments() == sorting0.get_num_segments() for sorting in sorting_list):
            raise Exception("Sorting objects must have the same number of segments.")

        # take sampling frequency from sorting list and test that they are equivalent.
        sampling_freqs = np.array([sorting.get_sampling_frequency() for sorting in sorting_list], dtype="float64")

        # Some sorter round the sampling freq lets emit a warning
        sf0 = sampling_freqs[0]
        if not np.all(sf0 == sampling_freqs):
            delta_freq_ratio = np.abs(sampling_freqs - sf0) / sf0
            # tolerance of 0.1%
            assert np.all(delta_freq_ratio < 0.001), "Inconsistent sampling frequency among sorting list"

        self.sampling_frequency = sf0
        self.delta_frames = int(self.delta_time / 1000 * self.sampling_frequency)


class MixinTemplateComparison:
    """
    Mixin for template comparisons to define:
       * similarity method
       * support
       * num_shifts
    """

    def __init__(self, similarity_method="cosine", support="union", num_shifts=0):
        self.similarity_method = similarity_method
        self.support = support
        self.num_shifts = num_shifts
