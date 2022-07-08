from .base import define_widget_function_from_class

from .unit_waveforms import UnitWaveformsWidget

class UnitTemplateWidget(UnitWaveformsWidget):
    possible_backends = {}


    def __init__(self, *args, **kargs):
        kargs['plot_waveforms'] = False
        UnitWaveformsWidget.__init__(self, *args, **kargs)


UnitTemplateWidget.__doc__ = UnitWaveformsWidget.__doc__

plot_unit_templates = define_widget_function_from_class(UnitTemplateWidget, 'plot_unit_templates')




