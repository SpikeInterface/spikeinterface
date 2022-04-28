from .mergesorting import MergeSorting
from .splitsortingunit import SplitSortingUnit
import networkx as nx
from collections import namedtuple
node_t = namedtuple('node_t', 'id stage')

class CurationSorting():
    """
    Class that handles curation of a Sorting object.

    """

    def __init__(self, parent_sorting):
        #to allow undo and redo a list of sortingextractors is keep
        self._sorting_stages = [parent_sorting]
        self._sorting_stages_i = 0
        parent_units = parent_sorting.get_unit_ids()
        #to easily allow undo and redo a list of graphs with the history of the curation is keep
        self._graphs = [nx.DiGraph()]
        self._graphs[0].add_nodes_from([node_t(u,0) for u in parent_units])

        self._kwargs = dict(parent_sorting=parent_sorting.to_dict())

    def _get_unused_id(self):
        #check units in the graph to the next unused unit id
        nodes = [n.id for n in self._graphs[self._sorting_stages_i].nodes()]
        return max(nodes)+1

    def split(self,unit2split, indices_list):
        current_sorting = self._sorting_stages[self._sorting_stages_i]
        new_id = self._get_unused_id()
        new_units_ids = [new_id, new_id+1]
        new_sorting = SplitSortingUnit(current_sorting, unit2split=unit2split, indices_list=indices_list, new_units_ids = new_units_ids)
        units = current_sorting.get_unit_ids()
        i = self._sorting_stages_i
        edges = [(node_t(u,i),node_t(u,i+1)) for u in units if u !=unit2split]
        edges = edges + [(node_t(unit2split,i), node_t(u,i+1)) for u in new_units_ids]
        self._add_new_stage(new_sorting, edges)

    def merge(self,units2merge, rm_dup_delta_time=0.4, cache=False):
        current_sorting = self._sorting_stages[self._sorting_stages_i]
        merge_unit_id = self._get_unused_id()
        new_sorting = MergeSorting(parent_sorting=current_sorting, units2merge=units2merge, 
                merge_unit_id=merge_unit_id, rm_dup_delta_time=rm_dup_delta_time, cache=cache)
        units = current_sorting.get_unit_ids()
        i = self._sorting_stages_i
        edges = [(node_t(u,i),node_t(u,i+1)) for u in units if u not in units2merge]
        edges = edges + [(node_t(u,i),node_t(merge_unit_id,i+1)) for u in units2merge]
        self._add_new_stage(new_sorting, edges)


    def remove_units(self, unit_ids):
        current_sorting = self._sorting_stages[self._sorting_stages_i]
        unit2keep = [u for u in current_sorting.get_unit_ids() if u not in unit_ids]
        i = self._sorting_stages_i
        edges = [(node_t(u,i),node_t(u,i+1)) for u in unit2keep]        
        self._add_new_stage(current_sorting.select_units(unit2keep),edges)

    def remove_unit(self, unit_id):
        self.remove_units([unit_id])

    def _add_new_stage(self, new_sorting, edges):
        #adds the stage to the stage list and creates the associated new graph
        self._sorting_stages = self._sorting_stages[0:self._sorting_stages_i+1]
        self._graphs = self._graphs[0:self._sorting_stages_i+1]
        self._sorting_stages.append(new_sorting)
        new_graph = self._graphs[self._sorting_stages_i].copy()
        new_graph.add_edges_from(edges)
        self._graphs.append(new_graph)
        self._sorting_stages_i += 1

    def select_units(self, unit_ids, renamed_unit_ids=None):
        self._sorting_stages = self._sorting_stages[0:self._sorting_stages_i]
        new_sorting = self._sorting_stages[-1].select_units(unit_ids, renamed_unit_ids)
        i = self._sorting_stages_i
        if renamed_unit_ids is None:
            edges = [(node_t(u,i), node_t(u,i+1)) for u in unit_ids] 
        else:
            edges = [(node_t(u,i), node_t(v,i+1)) for u,v in zip(unit_ids, renamed_unit_ids)] 
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

    def draw_graph(self):
        graph = self.graph
        pos = {n:(n.stage, -n.id) for n in graph.nodes}
        labels = {n:str(n.id) for n in graph.nodes}
        nx.draw(graph,pos,labels=labels)

    @property
    def graph(self):
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