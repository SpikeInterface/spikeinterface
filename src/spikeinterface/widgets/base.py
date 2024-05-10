from __future__ import annotations

import inspect

global default_backend_
default_backend_ = "matplotlib"

from ..core import SortingAnalyzer, BaseSorting
from ..core.waveforms_extractor_backwards_compatibility import MockWaveformExtractor


def get_default_plotter_backend():
    """Return the default backend for spikeinterface widgets.
    The default backend is "matplotlib" at init.
    It can be be globally set with `set_default_plotter_backend(backend)`
    """

    global default_backend_
    return default_backend_


def set_default_plotter_backend(backend):
    global default_backend_
    default_backend_ = backend


backend_kwargs_desc = {
    "matplotlib": {
        "figure": "Matplotlib figure. When None, it is created, default: None",
        "ax": "Single matplotlib axis. When None, it is created, default: None",
        "axes": "Multiple matplotlib axes. When None, they is created, default: None",
        "ncols": "Number of columns to create in subplots, default: 5",
        "figsize": "Size of matplotlib figure, default: None",
        "figtitle": "The figure title, default: None",
    },
    "sortingview": {
        "generate_url": "If True, the figurl URL is generated and printed, default: True",
        "display": "If True and in jupyter notebook/lab, the widget is displayed in the cell, default: True.",
        "figlabel": "The figurl figure label, default: None",
        "height": "The height of the sortingview View in jupyter, default: None",
    },
    "ipywidgets": {
        "width_cm": "Width of the figure in cm, default: 10",
        "height_cm": "Height of the figure in cm, default 6",
        "display": "If True, widgets are immediately displayed, default: True",
        # "controllers": ""
    },
    "ephyviewer": {},
    "spikeinterface_gui": {},
}

default_backend_kwargs = {
    "matplotlib": {"figure": None, "ax": None, "axes": None, "ncols": 5, "figsize": None, "figtitle": None},
    "sortingview": {"generate_url": True, "display": True, "figlabel": None, "height": None},
    "ipywidgets": {"width_cm": 25, "height_cm": 10, "display": True, "controllers": None},
    "ephyviewer": {},
    "spikeinterface_gui": {},
}


class BaseWidget:
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
            self.do_plot()

    # subclass must define one method per supported backend:
    # def plot_matplotlib(self, data_plot, **backend_kwargs):
    # def plot_ipywidgets(self, data_plot, **backend_kwargs):
    # def plot_sortingview(self, data_plot, **backend_kwargs):

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

    def do_plot(self):
        func = getattr(self, f"plot_{self.backend}")
        func(self.data_plot, **self.backend_kwargs)

    @classmethod
    def ensure_sorting_analyzer(cls, input):
        # internal help to accept both SortingAnalyzer or MockWaveformExtractor for a plotter
        if isinstance(input, SortingAnalyzer):
            return input
        elif isinstance(input, MockWaveformExtractor):
            return input.sorting_analyzer
        else:
            raise TypeError("input must be a SortingAnalyzer or MockWaveformExtractor")

    @classmethod
    def ensure_sorting(cls, input):
        # internal help to accept both Sorting or SortingAnalyzer or MockWaveformExtractor for a plotter
        if isinstance(input, BaseSorting):
            return input
        elif isinstance(input, SortingAnalyzer):
            return input.sorting
        elif isinstance(input, MockWaveformExtractor):
            return input.sorting_analyzer.sorting
        else:
            raise TypeError("input must be a SortingAnalyzer, MockWaveformExtractor, or of type BaseSorting")

    @staticmethod
    def check_extensions(sorting_analyzer, extensions):
        if isinstance(extensions, str):
            extensions = [extensions]
        error_msg = ""
        raise_error = False
        for extension in extensions:
            if not sorting_analyzer.has_extension(extension):
                raise_error = True
                error_msg += (
                    f"The {extension} waveform extension is required for this widget. "
                    f"Run the `sorting_analyzer.compute('{extension}', ...)` to compute it.\n"
                )
        if raise_error:
            raise Exception(error_msg)


class to_attr(object):
    def __init__(self, d):
        """
        Helper function that transform a dict into
        an object where attributes are the keys of the dict

        d = {"a": 1, "b": "yep"}
        o = to_attr(d)
        print(o.a, o.b)
        """
        object.__init__(self)
        object.__setattr__(self, "__d", d)

    def __getattribute__(self, k):
        d = object.__getattribute__(self, "__d")
        return d[k]
