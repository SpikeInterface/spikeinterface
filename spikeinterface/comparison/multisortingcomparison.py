import numpy as np
from pathlib import Path
import json
import os

from spikeinterface.core import load_extractor, BaseSorting, BaseSortingSegment
from .basecomparison import BaseComparison
from .symmetricsortingcomparison import SymmetricSortingComparison
from .comparisontools import compare_spike_trains

import networkx as nx


class MultiSortingComparison(BaseComparison):
    '''
    Compares multiple spike sorter outputs.

    - Pair-wise comparisons are made
    - An agreement graph is built based on the agreement score

    It allows to return a consensus-based sorting extractor with the `get_agreement_sorting()` method.

    Parameters
    ----------
    sorting_list: list
        List of sorting extractor objects to be compared
    name_list: list
        List of spike sorter names. If not given, sorters are named as 'sorter0', 'sorter1', 'sorter2', etc.
    delta_time: float
        Number of ms to consider coincident spikes (default 0.4 ms)
    match_score: float
        Minimum agreement score to match units (default 0.5)
    chance_score: float
        Minimum agreement score to for a possible match (default 0.1)
    n_jobs: int
       Number of cores to use in parallel. Uses all availible if -1
    spiketrain_mode: str
        Mode to extract agreement spike trains:
            - 'union': spike trains are the union between the spike trains of the best matching two sorters
            - 'intersection': spike trains are the intersection between the spike trains of the
               best matching two sorters
    verbose: bool
        if True, output is verbose

    Returns
    -------

    multi_sorting_comparison: MultiSortingComparison
        MultiSortingComparison object with the multiple sorter comparison
    '''

    def __init__(self, sorting_list, name_list=None, delta_time=0.4,  # sampling_frequency=None,
                 match_score=0.5, chance_score=0.1, n_jobs=-1, spiketrain_mode='union', verbose=False,
                 do_matching=True):
        BaseComparison.__init__(self, sorting_list, name_list=name_list,
                                delta_time=delta_time,  #  sampling_frequency=sampling_frequency,
                                match_score=match_score, chance_score=chance_score,
                                n_jobs=n_jobs, verbose=verbose)
        self._spiketrain_mode = spiketrain_mode
        self.clean_graph = None
        if do_matching:
            self._do_comparison()
            self._do_graph()
            self._remove_duplicate_edges()
            self._do_agreement()

    def get_agreement_sorting(self, minimum_agreement_count=1, minimum_agreement_count_only=False):
        '''
        Returns AgreementSortingExtractor with units with a 'minimum_matching' agreement.

        Parameters
        ----------
        minimum_agreement_count: int
            Minimum number of matches among sorters to include a unit.
        minimum_agreement_count_only: bool
            If True, only units with agreement == 'minimum_matching' are included.
            If False, units with an agreement >= 'minimum_matching' are included

        Returns
        -------
        agreement_sorting: AgreementSortingExtractor
            The output AgreementSortingExtractor
        '''
        assert minimum_agreement_count > 0, "'minimum_agreement_count' should be greater than 0"
        sorting = AgreementSortingExtractor(self.sampling_frequency, self,
                                            min_agreement_count=minimum_agreement_count,
                                            min_agreement_count_only=minimum_agreement_count_only)
        return sorting

    def compute_subgraphs(self):
        '''
        Computes subgraphs of connected components.

        Returns
        -------
        sg_sorter_names: list
            List of sorter names for each node in the connected component subrgaph
        sg_units: list
            List of unit ids for each node in the connected component subrgaph
        '''
        if self.clean_graph is not None:
            g = self.clean_graph
        else:
            g = self.graph
        subgraphs = (g.subgraph(c).copy() for c in nx.connected_components(g))
        sg_sorter_names = []
        sg_units = []
        for i, sg in enumerate(subgraphs):
            sorter_names = []
            sorter_units = []
            for node in sg.nodes:
                sorter_names.append(node[0])
                sorter_units.append(node[1])
            sg_sorter_names.append(sorter_names)
            sg_units.append(sorter_units)
        return sg_sorter_names, sg_units

    def dump(self, save_folder):
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        filename = str(save_folder / 'multicomparison.gpickle')
        nx.write_gpickle(self.graph, filename)
        kwargs = {'delta_time': self.delta_time, 'sampling_frequency': self.sampling_frequency,
                  'match_score': self.match_score, 'chance_score': self.chance_score,
                  'n_jobs': self._n_jobs, 'verbose': self._verbose}
        with (save_folder / 'kwargs.json').open('w') as f:
            json.dump(kwargs, f)
        sortings = {}
        for (name, sort) in zip(self.name_list, self.sorting_list):
            if sort.check_if_dumpable():
                sortings[name] = sort.make_serialized_dict()
            else:
                print(f'Skipping {name} because it is not dumpable')
        with (save_folder / 'sortings.json').open('w') as f:
            json.dump(sortings, f)

    @staticmethod
    def load_multicomparison(folder_path):
        folder_path = Path(folder_path)
        with (folder_path / 'kwargs.json').open() as f:
            kwargs = json.load(f)
        with (folder_path / 'sortings.json').open() as f:
            dict_sortings = json.load(f)
        name_list = dict_sortings.keys()
        sorting_list = [load_extractor(v) for v in dict_sortings.values()]
        mcmp = MultiSortingComparison(sorting_list=sorting_list, name_list=list(name_list), do_matching=False, **kwargs)
        mcmp.graph = nx.read_gpickle(str(folder_path / 'multicomparison.gpickle'))
        # do step 3 and 4
        mcmp._remove_duplicate_edges()
        mcmp._do_agreement()
        return mcmp

    def _do_comparison(self, ):
        # do pairwise matching
        if self._verbose:
            print('Multicomaprison step 1: pairwise comparison')

        self.comparisons = []
        for i in range(len(self.sorting_list)):
            for j in range(i + 1, len(self.sorting_list)):
                if self._verbose:
                    print("  Comparing: ", self.name_list[i], " and ", self.name_list[j])
                comp = SymmetricSortingComparison(self.sorting_list[i], self.sorting_list[j],
                                                  sorting1_name=self.name_list[i],
                                                  sorting2_name=self.name_list[j],
                                                  delta_time=self.delta_time,
                                                  # sampling_frequency=self.sampling_frequency,
                                                  match_score=self.match_score,
                                                  n_jobs=self._n_jobs,
                                                  verbose=False)
                self.comparisons.append(comp)

    def _do_graph(self):
        if self._verbose:
            print('Multicomaprison step 2: make graph')

        self.graph = nx.Graph()
        # nodes
        for i, sorting in enumerate(self.sorting_list):
            sorter_name = self.name_list[i]
            for unit_id in sorting.get_unit_ids():
                node = sorter_name, unit_id
                self.graph.add_node(node)

        # edges
        for comp in self.comparisons:
            for u1 in comp.sorting1.get_unit_ids():
                u2 = comp.hungarian_match_12[u1]
                if u2 != -1:
                    node1 = comp.sorting1_name, u1
                    node2 = comp.sorting2_name, u2
                    score = comp.agreement_scores.loc[u1, u2]
                    self.graph.add_edge(node1, node2, weight=score)

        # the graph is symmetrical
        self.graph = self.graph.to_undirected()

    def _do_agreement(self):
        # extract agreement from graph
        if self._verbose:
            print('Multicomaprison step 4: extract agreement from graph')

        self._new_units = {}
        self._spiketrains = []
        unit_id = 0

        # Remove duplicate
        subgraphs = (self.clean_graph.subgraph(c).copy() for c in nx.connected_components(self.clean_graph))
        for i, sg in enumerate(subgraphs):
            edges = list(sg.edges(data=True))
            if len(edges) > 0:
                avg_agr = np.mean([d['weight'] for u, v, d in edges])
            else:
                avg_agr = 0
            sorter_unit_ids = {}
            for node in sg.nodes:
                sorter_name = node[0]
                sorter_unit = node[1]
                sorter_unit_ids[sorter_name] = sorter_unit
            self._new_units[unit_id] = {'avg_agreement': avg_agr, 'sorter_unit_ids': sorter_unit_ids,
                                        'agreement_number': len(sg.nodes)}
            # Append correct spike train
            if len(sorter_unit_ids.keys()) == 1:
                self._spiketrains.append(self.sorting_list[self.name_list.index(
                    list(sorter_unit_ids.keys())[0])].get_unit_spike_train(list(sorter_unit_ids.values())[0]))
            else:
                max_edge = edges[int(np.argmax([d['weight'] for u, v, d in edges]))]
                node1, node2, weight = max_edge
                sorter1, unit1 = node1
                sorter2, unit2 = node2
                sp1 = self.sorting_list[self.name_list.index(sorter1)].get_unit_spike_train(unit1)
                sp2 = self.sorting_list[self.name_list.index(sorter2)].get_unit_spike_train(unit2)

                if self._spiketrain_mode == 'union':
                    lab1, lab2 = compare_spike_trains(sp1, sp2)
                    # add FP to spike train 1 (FP are the only spikes outside the union)
                    fp_idx2 = np.where(np.array(lab2) == 'FP')[0]
                    sp_union = np.sort(np.concatenate((sp1, sp2[fp_idx2])))
                    self._spiketrains.append(list(sp_union))
                elif self._spiketrain_mode == 'intersection':
                    lab1, lab2 = compare_spike_trains(sp1, sp2)
                    # TP are the spikes in the intersection
                    tp_idx1 = np.where(np.array(lab1) == 'TP')[0]
                    sp_tp1 = list(np.array(sp1)[tp_idx1])
                    self._spiketrains.append(sp_tp1)
            unit_id += 1

    def _do_agreement_matrix(self, minimum_agreement=1):
        sorted_name_list = sorted(self.name_list)
        sorting_agr = AgreementSortingExtractor(self.sampling_frequency, self, minimum_agreement)
        unit_ids = sorting_agr.get_unit_ids()
        agreement_matrix = np.zeros((len(unit_ids), len(sorted_name_list)))

        for u_i, unit in enumerate(unit_ids):
            for sort_name, sorter in enumerate(sorted_name_list):
                if sorter in sorting_agr.get_unit_property(unit, 'sorter_unit_ids').keys():
                    assigned_unit = sorting_agr.get_unit_property(unit, 'sorter_unit_ids')[sorter]
                else:
                    assigned_unit = -1
                if assigned_unit == -1:
                    agreement_matrix[u_i, sort_name] = np.nan
                else:
                    agreement_matrix[u_i, sort_name] = sorting_agr.get_unit_property(unit, 'avg_agreement')
        return agreement_matrix

    def _remove_duplicate_edges(self):
        if self._verbose:
            print('Multicomaprison step 3: clean graph')
        clean_graph = self.graph.copy()
        subgraphs = (clean_graph.subgraph(c).copy() for c in nx.connected_components(clean_graph))
        removed_nodes = 0
        for i, sg in enumerate(subgraphs):
            sorter_names = []
            for node in sg.nodes:
                sorter_names.append(node[0])
            sorters, counts = np.unique(sorter_names, return_counts=True)

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
                    remove_idxs = np.argsort(weights_duplicates)[:edges_to_remove]
                    for idx in remove_idxs:
                        if self._verbose:
                            print('Removed edge', edges_duplicates[idx])
                        clean_graph.remove_edge(edges_duplicates[idx][0], edges_duplicates[idx][1])
                        sg.remove_edge(edges_duplicates[idx][0], edges_duplicates[idx][1])
                        if edges_duplicates[idx][0] in nodes_duplicate:
                            sg.remove_node(edges_duplicates[idx][0])
                        else:
                            sg.remove_node(edges_duplicates[idx][1])
                        removed_nodes += 1
        if self._verbose:
            print(f'Removed {removed_nodes} duplicate nodes')
        self.clean_graph = clean_graph


class AgreementSortingExtractor(BaseSorting):

    def __init__(self, sampling_frequency, multisortingcomparison,
                 min_agreement_count=1, min_agreement_count_only=False):

        self._msc = multisortingcomparison
        self.is_dumpable = False

        # TODO: @alessio I leav this for you
        # if min_agreement_count_only:
        # self._unit_ids = list(u for u in self._msc._new_units.keys()
        # if self._msc._new_units[u]['agreement_number'] == min_agreement_count)
        # else:
        # self._unit_ids = list(u for u in self._msc._new_units.keys()
        # if self._msc._new_units[u]['agreement_number'] >= min_agreement_count)

        # for unit in self._unit_ids:
        # self.set_unit_property(unit_id=unit, property_name='agreement_number',
        # value=self._msc._new_units[unit]['agreement_number'])
        # self.set_unit_property(unit_id=unit, property_name='avg_agreement',
        # value=self._msc._new_units[unit]['avg_agreement'])
        # self.set_unit_property(unit_id=unit, property_name='sorter_unit_ids',
        # value=self._msc._new_units[unit]['sorter_unit_ids'])

        if min_agreement_count_only:
            unit_ids = list(u for u in self._msc._new_units.keys()
                            if self._msc._new_units[u]['agreement_number'] == min_agreement_count)
        else:
            unit_ids = list(u for u in self._msc._new_units.keys()
                            if self._msc._new_units[u]['agreement_number'] >= min_agreement_count)

        BaseSorting.__init__(self, sampling_frequency=sampling_frequency, unit_ids=unit_ids)
        if len(unit_ids) > 0:
            for k in ('agreement_number', 'avg_agreement', 'sorter_unit_ids'):
                values = [self._msc._new_units[unit_id][k] for unit_id in unit_ids]
                self.set_property(k, values, ids=unit_ids)

        sorting_segment = AgreementSortingSegment(multisortingcomparison)
        self.add_sorting_segment(sorting_segment)


class AgreementSortingSegment(BaseSortingSegment):
    def __init__(self, multisortingcomparison):
        BaseSortingSegment.__init__(self)
        self._msc = multisortingcomparison

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        ind = list(self._msc._new_units.keys()).index(unit_id)
        spiketrains = self._msc._spiketrains[ind]
        return np.asarray(spiketrains)


def compare_multiple_sorters(*args, **kwargs):
    return MultiSortingComparison(*args, **kwargs)


compare_multiple_sorters.__doc__ = MultiSortingComparison.__doc__
