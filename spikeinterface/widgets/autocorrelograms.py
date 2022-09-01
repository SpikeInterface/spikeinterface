from .crosscorrelograms import CrossCorrelogramsWidget


class AutoCorrelogramsWidget(CrossCorrelogramsWidget):
    possible_backends = {}

    def __init__(self, *args, **kargs):
        CrossCorrelogramsWidget.__init__(self, *args, **kargs)


AutoCorrelogramsWidget.__doc__ = CrossCorrelogramsWidget.__doc__






