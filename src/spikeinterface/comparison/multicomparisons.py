from __future__ import annotations

from pathlib import Path
import json
import pickle
import warnings

import numpy as np

from spikeinterface.core import load_extractor, BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class
from .basecomparison import BaseMultiComparison, MixinSpikeTrainComparison, MixinTemplateComparison
from .paircomparisons import SymmetricSortingComparison, TemplateComparison
from .comparisontools import compare_spike_trains


class MultiSortingComparison(BaseMultiComparison, MixinSpikeTrainComparison):
    """
    Compares multiple spike sorting outputs based on spike trains.

    - Pair-wise comparisons are made
    - An agreement graph is built based on the agreement score

    It allows to return a consensus-based sorting extractor with the `get_agreement_sorting()` method.

    Parameters
    ----------
    sorting_list : list
        List of sorting extractor objects to be compared
    name_list : list, default: None
        List of spike sorter names. If not given, sorters are named as "sorter0", "sorter1", "sorter2", etc.
    delta_time : float, default: 0.4
        Number of ms to consider coincident spikes
    match_score : float, default: 0.5
        Minimum agreement score to match units
    chance_score : float, default: 0.1
        Minimum agreement score to for a possible match
    n_jobs : int, default: -1
       Number of cores to use in parallel. Uses all available if -1
    spiketrain_mode : "union" | "intersection", default: "union"
        Mode to extract agreement spike trains:
            - "union" : spike trains are the union between the spike trains of the best matching two sorters
            - "intersection" : spike trains are the intersection between the spike trains of the
               best matching two sorters
    verbose : bool, default: False
        if True, output is verbose

    Returns
    -------
    multi_sorting_comparison : MultiSortingComparison
        MultiSortingComparison object with the multiple sorter comparison
    """

    def __init__(
        self,
        sorting_list,
        name_list=None,
        delta_time=0.4,  # sampling_frequency=None,
        match_score=0.5,
        chance_score=0.1,
        n_jobs=-1,
        spiketrain_mode="union",
        verbose=False,
        do_matching=True,
    ):
        if name_list is None:
            name_list = [f"sorting{i}" for i in range(len(sorting_list))]
        BaseMultiComparison.__init__(
            self,
            object_list=sorting_list,
            name_list=name_list,
            match_score=match_score,
            chance_score=chance_score,
            verbose=verbose,
        )
        MixinSpikeTrainComparison.__init__(self, delta_time=delta_time, n_jobs=n_jobs)
        self.set_frames_and_frequency(self.object_list)
        self._spiketrain_mode = spiketrain_mode
        self._spiketrains = None
        self._num_segments = sorting_list[0].get_num_segments()

        if do_matching:
            self._compute_all()
            self._populate_spiketrains()

    def _compare_ij(self, i, j):
        comp = SymmetricSortingComparison(
            self.object_list[i],
            self.object_list[j],
            sorting1_name=self.name_list[i],
            sorting2_name=self.name_list[j],
            delta_time=self.delta_time,
            match_score=self.match_score,
            n_jobs=self.n_jobs,
            verbose=False,
        )
        return comp

    def _populate_nodes(self):
        for i, sorting in enumerate(self.object_list):
            sorter_name = self.name_list[i]
            for unit_id in sorting.get_unit_ids():
                node = sorter_name, unit_id
                self.graph.add_node(node)

    def _populate_spiketrains(self):
        self._spiketrains = []
        for seg_index in range(self._num_segments):
            spike_trains_segment = dict()
            for unit_id, sg in zip(self._new_units, self.subgraphs):
                sorter_unit_ids = self._new_units[unit_id]["unit_ids"]
                edges = list(sg.edges(data=True))
                # Append correct spike train
                if len(sorter_unit_ids.keys()) == 1:
                    sorting = self.object_list[self.name_list.index(list(sorter_unit_ids.keys())[0])]
                    this_sorting_unit_id = list(sorter_unit_ids.values())[0]
                    spike_train = sorting.get_unit_spike_train(this_sorting_unit_id, seg_index)
                else:
                    max_edge = edges[int(np.argmax([d["weight"] for u, v, d in edges]))]
                    node1, node2, weight = max_edge
                    sorter1, unit1 = node1
                    sorter2, unit2 = node2

                    sorting1 = self.object_list[self.name_list.index(sorter1)]
                    sorting2 = self.object_list[self.name_list.index(sorter2)]
                    sp1 = sorting1.get_unit_spike_train(unit1, seg_index)
                    sp2 = sorting2.get_unit_spike_train(unit2, seg_index)
                    if self._spiketrain_mode == "union":
                        lab1, lab2 = compare_spike_trains(sp1, sp2)
                        # add FP to spike train 1 (FP are the only spikes outside the union)
                        fp_idx2 = np.where(np.array(lab2) == "FP")[0]
                        spike_train = np.sort(np.concatenate((sp1, sp2[fp_idx2])))
                    elif self._spiketrain_mode == "intersection":
                        lab1, lab2 = compare_spike_trains(sp1, sp2)
                        # TP are the spikes in the intersection
                        tp_idx1 = np.where(np.array(lab1) == "TP")[0]
                        spike_train = np.array(sp1)[tp_idx1]
                spike_trains_segment[unit_id] = spike_train
            self._spiketrains.append(spike_trains_segment)

    def _do_agreement_matrix(self, minimum_agreement=1):
        sorted_name_list = sorted(self.name_list)
        sorting_agr = AgreementSortingExtractor(self.sampling_frequency, self, minimum_agreement)
        unit_ids = sorting_agr.get_unit_ids()
        agreement_matrix = np.zeros((len(unit_ids), len(sorted_name_list)))

        for u_i, unit in enumerate(unit_ids):
            for sort_name, sorter in enumerate(sorted_name_list):
                if sorter in sorting_agr.get_unit_property(unit, "unit_ids").keys():
                    assigned_unit = sorting_agr.get_unit_property(unit, "unit_ids")[sorter]
                else:
                    assigned_unit = -1
                if assigned_unit == -1:
                    agreement_matrix[u_i, sort_name] = np.nan
                else:
                    agreement_matrix[u_i, sort_name] = sorting_agr.get_unit_property(unit, "avg_agreement")
        return agreement_matrix

    def get_agreement_sorting(self, minimum_agreement_count=1, minimum_agreement_count_only=False):
        """
        Returns AgreementSortingExtractor with units with a "minimum_matching" agreement.

        Parameters
        ----------
        minimum_agreement_count : int
            Minimum number of matches among sorters to include a unit.
        minimum_agreement_count_only : bool
            If True, only units with agreement == "minimum_matching" are included.
            If False, units with an agreement >= "minimum_matching" are included

        Returns
        -------
        agreement_sorting : AgreementSortingExtractor
            The output AgreementSortingExtractor
        """
        assert minimum_agreement_count > 0, "'minimum_agreement_count' should be greater than 0"
        sorting = AgreementSortingExtractor(
            self.sampling_frequency,
            self,
            min_agreement_count=minimum_agreement_count,
            min_agreement_count_only=minimum_agreement_count_only,
        )
        return sorting

    def save_to_folder(self, save_folder):
        warnings.warn(
            "save_to_folder() is deprecated. "
            "You should save and load the multi sorting comparison object using pickle."
            "\n>>> pickle.dump(mcmp, open('mcmp.pkl', 'wb')))))\n>>> mcmp_loaded = pickle.load(open('mcmp.pkl', 'rb'))",
            DeprecationWarning,
            stacklevel=2,
        )
        for sorting in self.object_list:
            assert sorting.check_serializability(
                "json"
            ), "MultiSortingComparison.save_to_folder() needs json serializable sortings"

        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        filename = str(save_folder / "multicomparison.gpickle")
        with open(filename, "wb") as f:
            pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)
        kwargs = {
            "delta_time": float(self.delta_time),
            "match_score": float(self.match_score),
            "chance_score": float(self.chance_score),
        }
        with (save_folder / "kwargs.json").open("w") as f:
            json.dump(kwargs, f)
        sortings = {}
        for name, sorting in zip(self.name_list, self.object_list):
            sortings[name] = sorting.to_dict(recursive=True, relative_to=save_folder)
        with (save_folder / "sortings.json").open("w") as f:
            json.dump(sortings, f)

    @staticmethod
    def load_from_folder(folder_path):
        warnings.warn(
            "load_from_folder() is deprecated. "
            "You should save and load the multi sorting comparison object using pickle."
            "\n>>> pickle.dump(mcmp, open('mcmp.pkl', 'wb')))))\n>>> mcmp_loaded = pickle.load(open('mcmp.pkl', 'rb'))",
            DeprecationWarning,
            stacklevel=2,
        )
        folder_path = Path(folder_path)
        with (folder_path / "kwargs.json").open() as f:
            kwargs = json.load(f)
        with (folder_path / "sortings.json").open() as f:
            dict_sortings = json.load(f)
        name_list = list(dict_sortings.keys())
        sorting_list = [load_extractor(v, base_folder=folder_path) for v in dict_sortings.values()]
        mcmp = MultiSortingComparison(sorting_list=sorting_list, name_list=list(name_list), do_matching=False, **kwargs)
        filename = str(folder_path / "multicomparison.gpickle")
        with open(filename, "rb") as f:
            mcmp.graph = pickle.load(f)
        # do step 3 and 4
        mcmp._clean_graph()
        mcmp._do_agreement()
        mcmp._populate_spiketrains()
        return mcmp


class AgreementSortingExtractor(BaseSorting):
    def __init__(
        self, sampling_frequency, multisortingcomparison, min_agreement_count=1, min_agreement_count_only=False
    ):
        self._msc = multisortingcomparison

        if min_agreement_count_only:
            unit_ids = list(
                u
                for u in self._msc._new_units.keys()
                if self._msc._new_units[u]["agreement_number"] == min_agreement_count
            )
        else:
            unit_ids = list(
                u
                for u in self._msc._new_units.keys()
                if self._msc._new_units[u]["agreement_number"] >= min_agreement_count
            )

        BaseSorting.__init__(self, sampling_frequency=sampling_frequency, unit_ids=unit_ids)

        self._serializability["json"] = False
        self._serializability["pickle"] = True

        if len(unit_ids) > 0:
            for k in ("agreement_number", "avg_agreement", "unit_ids"):
                values = [self._msc._new_units[unit_id][k] for unit_id in unit_ids]
                self.set_property(k, values, ids=unit_ids)

        for segment_index in range(multisortingcomparison._num_segments):
            sorting_segment = AgreementSortingSegment(multisortingcomparison._spiketrains[segment_index])
            self.add_sorting_segment(sorting_segment)

        self._kwargs = dict(
            sampling_frequency=sampling_frequency,
            multisortingcomparison=multisortingcomparison,
            min_agreement_count=min_agreement_count,
            min_agreement_count_only=min_agreement_count_only,
        )


class AgreementSortingSegment(BaseSortingSegment):
    def __init__(self, spiketrains_segment):
        BaseSortingSegment.__init__(self)
        self.spiketrains = spiketrains_segment

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        spiketrain = self.spiketrains[unit_id]
        if start_frame is not None:
            spiketrain = spiketrain[spiketrain >= start_frame]
        if end_frame is not None:
            spiketrain = spiketrain[spiketrain < end_frame]
        return spiketrain


compare_multiple_sorters = define_function_from_class(
    source_class=MultiSortingComparison, name="compare_multiple_sorters"
)


class MultiTemplateComparison(BaseMultiComparison, MixinTemplateComparison):
    """
    Compares multiple waveform extractors using template similarity.

    - Pair-wise comparisons are made
    - An agreement graph is built based on the agreement score

    Parameters
    ----------
    waveform_list : list
        List of waveform extractor objects to be compared
    name_list : list, default: None
        List of session names. If not given, sorters are named as "sess0", "sess1", "sess2", etc.
    match_score : float, default: 0.8
        Minimum agreement score to match units
    chance_score : float, default: 0.3
        Minimum agreement score to for a possible match
    verbose : bool, default: False
        if True, output is verbose

    Returns
    -------
    multi_template_comparison : MultiTemplateComparison
        MultiTemplateComparison object with the multiple template comparisons
    """

    def __init__(
        self,
        waveform_list,
        name_list=None,
        match_score=0.8,
        chance_score=0.3,
        verbose=False,
        similarity_method="cosine",
        support="union",
        num_shifts=0,
        do_matching=True,
    ):
        if name_list is None:
            name_list = [f"sess{i}" for i in range(len(waveform_list))]
        BaseMultiComparison.__init__(
            self,
            object_list=waveform_list,
            name_list=name_list,
            match_score=match_score,
            chance_score=chance_score,
            verbose=verbose,
        )
        MixinTemplateComparison.__init__(
            self, similarity_method=similarity_method, support=support, num_shifts=num_shifts
        )

        if do_matching:
            self._compute_all()

    def _compare_ij(self, i, j):
        comp = TemplateComparison(
            self.object_list[i],
            self.object_list[j],
            name1=self.name_list[i],
            name2=self.name_list[j],
            match_score=self.match_score,
            verbose=False,
        )
        return comp

    def _populate_nodes(self):
        for i, we in enumerate(self.object_list):
            session_name = self.name_list[i]
            for unit_id in we.unit_ids:
                node = session_name, unit_id
                self.graph.add_node(node)


compare_multiple_templates = define_function_from_class(
    source_class=MultiTemplateComparison, name="compare_multiple_templates"
)
