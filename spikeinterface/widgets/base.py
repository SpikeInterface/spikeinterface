import inspect

global default_backend_
default_backend_ = 'matplotlib'

def get_default_plotter_backend():
    """Return the default backend for spikeinterface widgets.
    The default backend is 'matplotlib' at init.
    It can be be globaly set with `set_default_plotter_backend(backend)`

    @jeremy: we could also used ENV variable if you prefer
    """

    global default_backend_
    return default_backend_


def set_default_plotter_backend(backend):
    global default_backend_
    default_backend_ = backend
    


class BaseWidget:
    # this need to be reset in the subclass
    possible_backends = None
    
    def __init__(self, plot_data=None, backend=None, **backend_kwargs):
        # every widgets must prepare a dict "plot_data" in the init
        self.plot_data = plot_data
        self.backend = backend
        
        # delegated to one of the plotter
        self.do_plot(backend, **backend_kwargs)

    def do_plot(self, backend, **backend_kwargs):
        if backend is None:
            backend = get_default_plotter_backend()
    
        assert backend in self.possible_backends, f'Backend {backend} not supported for this widget'    
        plotter = self.possible_backends[backend]()
        plotter.do_plot(self.plot_data, **backend_kwargs)

    @classmethod
    def register_backend(cls, backend_plotter):
        cls.possible_backends[backend_plotter.backend] = backend_plotter   


class BackendPlotter():
    backend = ''
    
    @classmethod
    def register(cls, widget_cls):
        #~ print('BackendPlotter.register', print(cls), isinstance(cls, BackendPlotter))
        widget_cls.register_backend(cls)

    def update_backend_kwargs(self, **backend_kwargs):
        backend_kwargs_ = self.default_backend_kwargs.copy()
        backend_kwargs_.update(backend_kwargs)
        return backend_kwargs_
    
def copy_signature(source_fct):
    def copy(target_fct):
        target_fct.__signature__ = inspect.signature(source_fct)
        return target_fct
    return copy


def define_widget_function_from_class(widget_class, name):

    @copy_signature(widget_class)
    def widget_func(*args, **kwargs):
        W = widget_class(*args, **kwargs)
        # @alessio @jeremy @jeff
        # function return the widget itself
        # we could return something else if needed  as we discussed
        return W

    widget_func.__doc__ = widget_class.__doc__
    widget_func.__name__ = name

    return widget_func
