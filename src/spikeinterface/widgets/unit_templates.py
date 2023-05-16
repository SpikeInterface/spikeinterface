from .unit_waveforms import UnitWaveformsWidget

class UnitTemplatesWidget(UnitWaveformsWidget):
    possible_backends = {}


    def __init__(self, *args, **kargs):
        kargs['plot_waveforms'] = False
        UnitWaveformsWidget.__init__(self, *args, **kargs)


UnitTemplatesWidget.__doc__ = UnitWaveformsWidget.__doc__






