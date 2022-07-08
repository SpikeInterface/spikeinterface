from ...core.core_tools import check_json
from spikeinterface.widgets.base import BackendPlotter

class SortingviewPlotter(BackendPlotter):
    backend = 'sortingview'
    backend_kwargs_desc = {
        "generate_url": "If True, the figurl URL is generated and printed. Default is True"
    }
    default_backend_kwargs = {
        "generate_url": True
    }
    
    def make_serializable(*args):
        serializable_dict = check_json({i: a for i, a in enumerate(args[1:])})
        returns = ()
        for i in range(len(args) - 1):
            returns += (serializable_dict[i],)
        return returns
        
