from .mergesorting import MergeSorting
from .splitsortingunit import SplitSortingUnit
from collections import namedtuple
import numpy as np
node_t = namedtuple('node_t', 'id stage')

class CurationSorting():
    """
    Class that handles curation of a Sorting object.

    Parameters
    ----------
    parent_sorting: Recording
        The recording object
    make_graph: bool
        True to keep a networksx graph with the curation history
    Returns
    -------
    sorting: Sorting
        Sorting object with the selected units merged
    """


    def __init__(self, parent_sorting, make_graph=False):
        #to allow undo and redo a list of sortingextractors is keep
        self._sorting_stages = [parent_sorting]
        self._sorting_stages_i = 0
        parent_units = parent_sorting.get_unit_ids()
        self._make_graph = make_graph
        if make_graph:
            #to easily allow undo and redo a list of graphs with the history of the curation is keep
            import networkx as nx
            self._nx = nx
            self._graphs = [self._nx.DiGraph()]
            self._graphs[0].add_nodes_from([node_t(u,0) for u in parent_units])
        # check the maximum numeric id used, strings with digits will be casted to int to reduce confusion
        if np.issubdtype(parent_units.dtype, np.character):
            self.max_used_id = max([-1]+[int(p) for p in parent_units if p.isdigit()])
        else:
            self.max_used_id = max(parent_units)
        
        self._kwargs = dict(parent_sorting=parent_sorting.to_dict())

    def _get_unused_id(self, n=1):
        #check units in the graph to the next unused unit id
        ids =[self.max_used_id+i for i in range(1,1+n)]
        if  np.issubdtype(self.sorting.get_unit_ids().dtype, np.character):
            ids = [str(i) for i in ids]
        return ids

    def split(self,unit2split, indices_list):
        current_sorting = self._sorting_stages[self._sorting_stages_i]
        new_units_ids = self._get_unused_id(2)
        new_sorting = SplitSortingUnit(current_sorting, unit2split=unit2split, indices_list=indices_list, new_units_ids = new_units_ids)

        if self._make_graph:
            units = current_sorting.get_unit_ids()
            i = self._sorting_stages_i
            edges = [(node_t(u,i),node_t(u,i+1)) for u in units if u !=unit2split]
            edges = edges + [(node_t(unit2split,i), node_t(u,i+1)) for u in new_units_ids]
        else:
            edges = None
        self.max_used_id = self.max_used_id+2
        self._add_new_stage(new_sorting, edges)

    def merge(self,units2merge, delta_time=0.4, cache=False):
        current_sorting = self._sorting_stages[self._sorting_stages_i]
        merge_unit_id = self._get_unused_id()[0]
        new_sorting = MergeSorting(parent_sorting=current_sorting, units2merge=units2merge, 
                merge_unit_id=merge_unit_id, delta_time=delta_time, cache=cache)
        if self._make_graph:
            units = current_sorting.get_unit_ids()
            i = self._sorting_stages_i
            edges = [(node_t(u,i),node_t(u,i+1)) for u in units if u not in units2merge]
            edges = edges + [(node_t(u,i),node_t(merge_unit_id,i+1)) for u in units2merge]
        else:
            edges = None
        self.max_used_id = self.max_used_id + 1
        self._add_new_stage(new_sorting, edges)


    def remove_units(self, unit_ids):
        current_sorting = self._sorting_stages[self._sorting_stages_i]
        unit2keep = [u for u in current_sorting.get_unit_ids() if u not in unit_ids]
        if self._make_graph:
            i = self._sorting_stages_i
            edges = [(node_t(u,i),node_t(u,i+1)) for u in unit2keep]
        else:
            edges = None
        self._add_new_stage(current_sorting.select_units(unit2keep),edges)

    def remove_unit(self, unit_id):
        self.remove_units([unit_id])

    def _add_new_stage(self, new_sorting, edges):
        #adds the stage to the stage list and creates the associated new graph
        self._sorting_stages = self._sorting_stages[0:self._sorting_stages_i+1]
        self._sorting_stages.append(new_sorting)
        if self._make_graph:
            self._graphs = self._graphs[0:self._sorting_stages_i+1]
            new_graph = self._graphs[self._sorting_stages_i].copy()
            new_graph.add_edges_from(edges)
            self._graphs.append(new_graph)
        self._sorting_stages_i += 1

    def select_units(self, unit_ids, renamed_unit_ids=None):
        new_sorting = self._sorting_stages[self._sorting_stages_i].select_units(unit_ids, renamed_unit_ids)
        if self._make_graph:
            i = self._sorting_stages_i
            if renamed_unit_ids is None:
                edges = [(node_t(u,i), node_t(u,i+1)) for u in unit_ids] 
            else:
                edges = [(node_t(u,i), node_t(v,i+1)) for u,v in zip(unit_ids, renamed_unit_ids)] 
        else:
            edges = None        
        self._add_new_stage(new_sorting, edges)

    def remove_empty_units(self):
        i = self._sorting_stages_i
        new_sorting = self._sorting_stages[i].remove_empty_units()
        if self._make_graph:
            curr_ids = self._sorting_stages[i].get_unit_ids()
            edges = [(node_t(u,i), node_t(u,i+1)) for u in curr_ids] 
        else:
            edges = None        
        self._add_new_stage(new_sorting, edges)        

    def redo_avaiable(self):
        #usefull function for a gui
        return self._sorting_stages_i < len(self._sorting_stages)
    
    def undo_avaiable(self):
        #usefull function for a gui
        return self._sorting_stages_i > 0

    def undo(self):
        if self.undo_avaiable():
            self._sorting_stages_i -=1
    
    def redo(self):
        if self.redo_avaiable():
            self._sorting_stages_i +=1

    def draw_graph(self, **kwargs):
        assert self._make_graph, 'to make graph make_graph=True'
        graph = self.graph
        ids = [c.id for c in graph.nodes]
        pos = {n:(n.stage, -ids.index(n.id)) for n in graph.nodes}
        labels = {n:str(n.id) for n in graph.nodes}
        self._nx.draw(graph, pos=pos, labels=labels, **kwargs)

    @property
    def graph(self):
        assert self._make_graph, 'to have a graph make_graph=True'
        return self._graphs[self._sorting_stages_i]


    @property
    def sorting(self):
        return self._sorting_stages[self._sorting_stages_i]

    def __getattr__(self,name):
        #any method not define for this class will try to use the current 
        #  sorting stage. In that whay this class will behave as a sortingextractor
        current_sorting = self._sorting_stages[self._sorting_stages_i]

        attr = object.__getattribute__(current_sorting, name)
        return attr