from .base import define_widget_function_from_class

from .crosscorrelograms import CrossCorrelogramsWidget


class AutoCorrelogramsWidget(CrossCorrelogramsWidget):
    possible_backends = {}

    def __init__(self, *args, **kargs):
        CrossCorrelogramsWidget.__init__(self, *args, **kargs)


AutoCorrelogramsWidget.__doc__ = CrossCorrelogramsWidget.__doc__

plot_autocorrelograms = define_widget_function_from_class(AutoCorrelogramsWidget, 'plot_autocorrelograms')




