import numpy as np
from copy import deepcopy

from ...core.core_tools import check_json
from spikeinterface.widgets.base import BackendPlotter


class SortingviewPlotter(BackendPlotter):
    backend = 'sortingview'
    backend_kwargs_desc = {
        "generate_url": "If True, the figurl URL is generated and printed. Default True",
        "display": "If True and in jupyter notebook/lab, the widget is displayed in the cell. Default True.",
        "figlabel": "The figurl figure label. Default None",
        "height": "The height of the sortingview View in jupyter. Default None"
    }
    default_backend_kwargs = {
        "generate_url": True,
        "display": True,
        "figlabel": None,
        "height": None
    }
    
    def __init__(self):
        self.view = None
        self.url = None

    def make_serializable(*args):
        serializable_dict = check_json({i: a for i, a in enumerate(args[1:])})
        returns = ()
        for i in range(len(args) - 1):
            returns += (serializable_dict[i],)
        if len(returns) == 1:
            returns = returns[0]
        return returns


    @staticmethod
    def is_notebook() -> bool:
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False

    def handle_display_and_url(self, view, **backend_kwargs):
        self.set_view(view)
        if self.is_notebook() and backend_kwargs["display"]:
            display(self.view.jupyter(height=backend_kwargs["height"]))
        if backend_kwargs["generate_url"]:
            figlabel = backend_kwargs.get("figlabel")
            if figlabel is None:
                figlabel = self.default_label
            url = view.url(label=figlabel)
            self.set_url(url)
            print(url)            

    # make view and url accessible by the plotter
    def set_view(self, view):
        self.view = view

    def set_url(self, url):
        self.url = url


def generate_unit_table_view(sorting, unit_properties=None):
    import sortingview.views as vv
    if unit_properties is None:
        ut_columns = []
        ut_rows = [
            vv.UnitsTableRow(unit_id=u, values={})
            for u in sorting.unit_ids
        ]
    else:
        ut_columns = unit_properties
        ut_rows = []
        values = {}
        for prop_name in unit_properties:
            property_values = sorting.get_property(prop_name)
            for ui, unit in enumerate(sorting.unit_ids):
                if np.isnan(property_values[ui]):
                    continue
                values[unit] = property_values[ui]
            
    v_units_table = vv.UnitsTable(rows=ut_rows, columns=ut_columns)
    return v_units_table
