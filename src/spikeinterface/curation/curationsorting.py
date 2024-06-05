from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable

import numpy as np

from .mergeunitssorting import MergeUnitsSorting
from .splitunitsorting import SplitUnitSorting
from spikeinterface.core.core_tools import define_function_from_class

node_t = namedtuple("node_t", "unit_id stage_id")


class CurationSorting:
    """
    Class that handles curation of a Sorting object.

    Parameters
    ----------
    sorting: BaseSorting
        The sorting object
    properties_policy : "keep" | "remove", default: "keep"
        Policy used to propagate properties after split and merge operation. If "keep" the properties will be
        passed to the new units (if the original units have the same value). If "remove" the new units will have
        an empty value for all the properties
    make_graph : bool
        True to keep a Networkx graph instance with the curation history
    Returns
    -------
    sorting : Sorting
        Sorting object with the selected units merged
    """

    def __init__(self, sorting, make_graph=False, properties_policy="keep"):

        # to allow undo and redo a list of sortingextractors is keep
        self._sorting_stages = [sorting]
        self._sorting_stages_i = 0
        self._properties_policy = properties_policy
        parent_units = sorting.get_unit_ids()
        self._make_graph = make_graph
        if make_graph:
            # to easily allow undo and redo a list of graphs with the history of the curation is keep
            import networkx as nx

            self._nx = nx
            self._graphs = [self._nx.DiGraph()]
            self._graphs[0].add_nodes_from([node_t(u, 0) for u in parent_units])
        # check the maximum numeric id used, strings with digits will be cast to int to reduce confusion
        if np.issubdtype(parent_units.dtype, np.character):
            self.max_used_id = max([-1] + [int(p) for p in parent_units if p.isdigit()])
        else:
            self.max_used_id = max(parent_units) if len(parent_units) > 0 else 0

        self._kwargs = dict(sorting=sorting, make_graph=make_graph, properties_policy=properties_policy)

    def _get_unused_id(self, n=1):
        # check units in the graph to the next unused unit id
        ids = [self.max_used_id + i for i in range(1, 1 + n)]
        if np.issubdtype(self.sorting.get_unit_ids().dtype, np.character):
            ids = [str(i) for i in ids]
        return ids

    def split(self, split_unit_id, indices_list, new_unit_ids=None):
        """
        Split a unit into multiple units.

        Parameters
        ----------
        split_unit_id : int or str
            The unit to split
        indices_list : list or np.array
            A list of index arrays selecting the spikes to split in each segment.
            Each array can contain more than 2 indices (e.g. for splitting in 3 or more units) and it should
            be the same length as the spike train (for each segment).
            If the sorting has only one segment, indices_list can be a single array
        new_unit_ids : list[str|int] ot None
            List of new unit ids. If None, a new unit id is automatically selected
        """
        current_sorting = self._sorting_stages[self._sorting_stages_i]
        if not isinstance(indices_list, list):
            indices_list = [indices_list]
        if not isinstance(indices_list[0], Iterable):
            raise ValueError("indices_list must be a list of iterable arrays")
        if new_unit_ids is None:
            new_unit_ids = self._get_unused_id(np.max([len(np.unique(v)) for v in indices_list]))
        new_sorting = SplitUnitSorting(
            current_sorting,
            split_unit_id=split_unit_id,
            indices_list=indices_list,
            new_unit_ids=new_unit_ids,
            properties_policy=self._properties_policy,
        )

        if self._make_graph:
            units = current_sorting.get_unit_ids()
            i = self._sorting_stages_i
            edges = [(node_t(u, i), node_t(u, i + 1)) for u in units if u != split_unit_id]
            edges = edges + [(node_t(split_unit_id, i), node_t(u, i + 1)) for u in new_unit_ids]
        else:
            edges = None
        self.max_used_id = self.max_used_id + 2
        self._add_new_stage(new_sorting, edges)

    def merge(self, units_to_merge, new_unit_id=None, delta_time_ms=0.4):
        """
        Merge a list of units into a new unit.

        Parameters
        ----------
        units_to_merge : list[str|int]
            List of unit ids to merge
        new_unit_id : int or str
            The new unit id. If None, a new unit id is automatically selected
        delta_time_ms : float
            Number of ms to consider for duplicated spikes. None won't check for duplications
        """
        current_sorting = self._sorting_stages[self._sorting_stages_i]
        if new_unit_id is None:
            new_unit_id = self._get_unused_id()[0]
        elif new_unit_id not in units_to_merge:
            assert new_unit_id not in current_sorting.unit_ids, f"new_unit_id already exists!"
        new_sorting = MergeUnitsSorting(
            sorting=current_sorting,
            units_to_merge=units_to_merge,
            new_unit_ids=[new_unit_id],
            delta_time_ms=delta_time_ms,
            properties_policy=self._properties_policy,
        )
        if self._make_graph:
            units = current_sorting.get_unit_ids()
            i = self._sorting_stages_i
            edges = [(node_t(u, i), node_t(u, i + 1)) for u in units if u not in units_to_merge]
            edges = edges + [(node_t(u, i), node_t(new_unit_id, i + 1)) for u in units_to_merge]
        else:
            edges = None
        self.max_used_id = self.max_used_id + 1
        self._add_new_stage(new_sorting, edges)

    def remove_units(self, unit_ids):
        """
        Remove a list of units.

        Parameters
        ----------
        unit_ids : list[str|int]
            List of unit ids to remove
        """
        current_sorting = self._sorting_stages[self._sorting_stages_i]
        unit2keep = [u for u in current_sorting.get_unit_ids() if u not in unit_ids]
        if self._make_graph:
            i = self._sorting_stages_i
            edges = [(node_t(u, i), node_t(u, i + 1)) for u in unit2keep]
        else:
            edges = None
        self._add_new_stage(current_sorting.select_units(unit2keep), edges)

    def remove_unit(self, unit_id):
        """
        Remove a unit.

        Parameters
        ----------
        unit_id : int ot str
            The unit id to remove
        """
        self.remove_units([unit_id])

    def select_units(self, unit_ids, renamed_unit_ids=None):
        """
        Select a list of units.

        Parameters
        ----------
        unit_ids : list[str|int]
            List of unit ids to select
        renamed_unit_ids : list or None, default: None
            List of new unit ids to rename the selected units
        """
        new_sorting = self._sorting_stages[self._sorting_stages_i].select_units(unit_ids, renamed_unit_ids)
        if self._make_graph:
            i = self._sorting_stages_i
            if renamed_unit_ids is None:
                edges = [(node_t(u, i), node_t(u, i + 1)) for u in unit_ids]
            else:
                edges = [(node_t(u, i), node_t(v, i + 1)) for u, v in zip(unit_ids, renamed_unit_ids)]
        else:
            edges = None
        self._add_new_stage(new_sorting, edges)

    def rename(self, renamed_unit_ids):
        """
        Rename a list of units.

        Parameters
        ----------
        renamed_unit_ids : list[str|int]
            List of unit ids to rename exisiting units
        """
        self.select_units(self.current_sorting.unit_ids, renamed_unit_ids=renamed_unit_ids)

    def remove_empty_units(self):
        """
        Remove empty units.
        """
        i = self._sorting_stages_i
        new_sorting = self._sorting_stages[i].remove_empty_units()
        if self._make_graph:
            curr_ids = self._sorting_stages[i].get_unit_ids()
            edges = [(node_t(u, i), node_t(u, i + 1)) for u in curr_ids]
        else:
            edges = None
        self._add_new_stage(new_sorting, edges)

    def redo_available(self):
        """
        Check if redo is available.

        Returns
        -------
        bool
            True if redo is available
        """
        # useful function for a gui
        return self._sorting_stages_i < len(self._sorting_stages)

    def undo_available(self):
        """
        Check if undo is available.

        Returns
        -------
        bool
            True if undo is available
        """
        # useful function for a gui
        return self._sorting_stages_i > 0

    def undo(self):
        """
        Undo the last operation.
        """
        if self.undo_available():
            self._sorting_stages_i -= 1

    def redo(self):
        """
        Redo the last operation.
        """
        if self.redo_available():
            self._sorting_stages_i += 1

    def draw_graph(self, **kwargs):
        """
        Draw the curation graph.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for Networkx draw function
        """
        assert self._make_graph, "to make a graph use make_graph=True"
        graph = self.graph
        ids = [c.unit_id for c in graph.nodes]
        pos = {n: (n.stage_id, -ids.index(n.unit_id)) for n in graph.nodes}
        labels = {n: str(n.unit_id) for n in graph.nodes}
        self._nx.draw(graph, pos=pos, labels=labels, **kwargs)

    @property
    def graph(self):
        assert self._make_graph, "to have a graph use make_graph=True"
        return self._graphs[self._sorting_stages_i]

    @property
    def sorting(self):
        return self.current_sorting

    @property
    def current_sorting(self):
        return self._sorting_stages[self._sorting_stages_i]

    def _add_new_stage(self, new_sorting, edges):
        # adds the stage to the stage list and creates the associated new graph
        self._sorting_stages = self._sorting_stages[0 : self._sorting_stages_i + 1]
        self._sorting_stages.append(new_sorting)
        if self._make_graph:
            self._graphs = self._graphs[0 : self._sorting_stages_i + 1]
            new_graph = self._graphs[self._sorting_stages_i].copy()
            new_graph.add_edges_from(edges)
            self._graphs.append(new_graph)
        self._sorting_stages_i += 1


curation_sorting = define_function_from_class(source_class=CurationSorting, name="curation_sorting")
