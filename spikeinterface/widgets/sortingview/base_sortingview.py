from ...core.core_tools import check_json
from spikeinterface.widgets.base import BackendPlotter

class SortingviewPlotter(BackendPlotter):
    backend = 'sortingview'
    backend_kwargs_desc = {
        "generate_url": "If True, the figurl URL is generated and printed. Default is True",
        "figlabel": "The figurl figure label. Default None"
    }
    default_backend_kwargs = {
        "generate_url": True,
        "figlabel": None
    }
    
    def make_serializable(*args):
        serializable_dict = check_json({i: a for i, a in enumerate(args[1:])})
        returns = ()
        for i in range(len(args) - 1):
            returns += (serializable_dict[i],)
        if len(returns) == 1:
            returns = returns[0]
        return returns

    def set_view(self, view):
        self.view = view


def generate_unit_table_view(unit_ids):
    import sortingview.views as vv
    ut_rows = [
        vv.UnitsTableRow(unit_id=u, values={})
        for u in unit_ids
    ]
    ut_columns = []
    v_units_table = vv.UnitsTable(rows=ut_rows, columns=ut_columns)
    return v_units_table
