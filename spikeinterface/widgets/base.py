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
        self.backend_kwargs = backend_kwargs

        
    def check_backend(self, backend, **backend_kwargs):
        if backend is None:
            backend = get_default_plotter_backend()
        assert backend in self.possible_backends, (f"{backend} backend not available! Available backends are: "
                                                   f"{list(self.possible_backends.keys())}")
        return backend, backend_kwargs

    def do_plot(self, backend, **backend_kwargs):
        backend, backend_kwargs = self.check_backend(backend, **backend_kwargs)
        plotter = self.possible_backends[backend]()
        plotter.do_plot(self.plot_data, **backend_kwargs)
        self.plotter = plotter

    @classmethod
    def register_backend(cls, backend_plotter):
        cls.possible_backends[backend_plotter.backend] = backend_plotter   

    @staticmethod
    def check_extensions(waveform_extractor, extensions):
        if isinstance(extensions, str):
            extensions = [extensions]
        error_msg = ""
        raise_error = False
        for extension in extensions:
            if not waveform_extractor.is_extension(extension):
                raise_error = True
                error_msg += f"The {extension} waveform extension is required for this widget. " \
                             f"Run the `compute_{extension}` to compute it.\n"
        if raise_error:
            raise Exception(error_msg)


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


class to_attr(object):
    def __init__(self, d):
        """
        Helper function that transform a dict into
        an object where attributes are the keys of the dict

        d = {'a': 1, 'b': 'yep'}
        o = to_attr(d)
        print(o.a, o.b)
        """
        object.__init__(self)
        object.__setattr__(self, '__d', d)

    def __getattribute__(self, k):
        d = object.__getattribute__(self, '__d')
        return d[k]

def define_widget_function_from_class(widget_class, name):

    @copy_signature(widget_class)
    def widget_func(*args, **kwargs):
        W = widget_class(*args, **kwargs)
        W.do_plot(W.backend, **W.backend_kwargs)
        return W.plotter

    widget_func.__doc__ = widget_class.__doc__
    widget_func.__name__ = name

    return widget_func
