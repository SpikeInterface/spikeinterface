import inspect

global default_backend_
default_backend_ = "matplotlib"


def get_default_plotter_backend():
    """Return the default backend for spikeinterface widgets.
    The default backend is 'matplotlib' at init.
    It can be be globally set with `set_default_plotter_backend(backend)`
    """

    global default_backend_
    return default_backend_


def set_default_plotter_backend(backend):
    global default_backend_
    default_backend_ = backend


backend_kwargs_desc = {
    "matplotlib": {
        "figure": "Matplotlib figure. When None, it is created. Default None",
        "ax": "Single matplotlib axis. When None, it is created. Default None",
        "axes": "Multiple matplotlib axes. When None, they is created. Default None",
        "ncols": "Number of columns to create in subplots.  Default 5",
        "figsize": "Size of matplotlib figure. Default None",
        "figtitle": "The figure title. Default None",
    },
    "sortingview": {
        "generate_url": "If True, the figurl URL is generated and printed. Default True",
        "display": "If True and in jupyter notebook/lab, the widget is displayed in the cell. Default True.",
        "figlabel": "The figurl figure label. Default None",
        "height": "The height of the sortingview View in jupyter. Default None",
    },
    "ipywidgets": {
        "width_cm": "Width of the figure in cm (default 10)",
        "height_cm": "Height of the figure in cm (default 6)",
        "display": "If True, widgets are immediately displayed",
    },
}

default_backend_kwargs = {
    "matplotlib": {"figure": None, "ax": None, "axes": None, "ncols": 5, "figsize": None, "figtitle": None},
    "sortingview": {"generate_url": True, "display": True, "figlabel": None, "height": None},
    "ipywidgets": {"width_cm": 25, "height_cm": 10, "display": True},
}


class BaseWidget:
    # this need to be reset in the subclass
    possible_backends = None

    def __init__(
        self,
        data_plot=None,
        backend=None,
        immediate_plot=True,
        **backend_kwargs,
    ):
        # every widgets must prepare a dict "plot_data" in the init
        self.data_plot = data_plot
        backend = self.check_backend(backend)
        self.backend = backend

        # check backend kwargs
        for k in backend_kwargs:
            if k not in default_backend_kwargs[backend]:
                raise Exception(
                    f"{k} is not a valid plot argument or backend keyword argument. "
                    f"Possible backend keyword arguments for {backend} are: {list(default_backend_kwargs[backend].keys())}"
                )
        backend_kwargs_ = default_backend_kwargs[self.backend].copy()
        backend_kwargs_.update(backend_kwargs)

        self.backend_kwargs = backend_kwargs_

        if immediate_plot:
            print("immediate_plot", self.backend, self.backend_kwargs)
            self.do_plot(self.backend, **self.backend_kwargs)

    @classmethod
    def get_possible_backends(cls):
        return [k for k in default_backend_kwargs if hasattr(cls, f"plot_{k}")]

    def check_backend(self, backend):
        if backend is None:
            backend = get_default_plotter_backend()
        assert backend in self.get_possible_backends(), (
            f"{backend} backend not available! Available backends are: " f"{self.get_possible_backends()}"
        )
        return backend

    # def check_backend_kwargs(self, plotter, backend, **backend_kwargs):
    #     plotter_kwargs = plotter.default_backend_kwargs
    #     for k in backend_kwargs:
    #         if k not in plotter_kwargs:
    #             raise Exception(
    #                 f"{k} is not a valid plot argument or backend keyword argument. "
    #                 f"Possible backend keyword arguments for {backend} are: {list(plotter_kwargs.keys())}"
    #             )

    def do_plot(self, backend, **backend_kwargs):
        # backend = self.check_backend(backend)

        func = getattr(self, f"plot_{backend}")
        func(self.data_plot, **self.backend_kwargs)

    # @classmethod
    # def register_backend(cls, backend_plotter):
    #     cls.possible_backends[backend_plotter.backend] = backend_plotter

    @staticmethod
    def check_extensions(waveform_extractor, extensions):
        if isinstance(extensions, str):
            extensions = [extensions]
        error_msg = ""
        raise_error = False
        for extension in extensions:
            if not waveform_extractor.is_extension(extension):
                raise_error = True
                error_msg += (
                    f"The {extension} waveform extension is required for this widget. "
                    f"Run the `compute_{extension}` to compute it.\n"
                )
        if raise_error:
            raise Exception(error_msg)


# class BackendPlotter:
#     backend = ""

#     @classmethod
#     def register(cls, widget_cls):
#         widget_cls.register_backend(cls)

#     def update_backend_kwargs(self, **backend_kwargs):
#         backend_kwargs_ = self.default_backend_kwargs.copy()
#         backend_kwargs_.update(backend_kwargs)
#         return backend_kwargs_


# def copy_signature(source_fct):
#     def copy(target_fct):
#         target_fct.__signature__ = inspect.signature(source_fct)
#         return target_fct

#     return copy


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
        object.__setattr__(self, "__d", d)

    def __getattribute__(self, k):
        d = object.__getattribute__(self, "__d")
        return d[k]


# def define_widget_function_from_class(widget_class, name):
#     @copy_signature(widget_class)
#     def widget_func(*args, **kwargs):
#         W = widget_class(*args, **kwargs)
#         W.do_plot(W.backend, **W.backend_kwargs)
#         return W.plotter

#     widget_func.__doc__ = widget_class.__doc__
#     widget_func.__name__ = name

#     return widget_func
