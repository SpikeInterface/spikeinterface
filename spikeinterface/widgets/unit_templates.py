import numpy as np

from .base import BaseWidget, define_widget_function_from_class
from ..core.waveform_extractor import WaveformExtractor
from ..core.baserecording import BaseRecording
from ..core.basesorting import BaseSorting
from .utils import get_unit_colors
from ..toolkit import get_template_channel_sparsity


from .unit_waveforms import UnitWaveformsWidget

class UnitTemplateWidget(UnitWaveformsWidget):
    possible_backends = {}

    def __init__(self, *args, **kargs):
        kargs['plot_waveforms'] = False
        UnitWaveformsWidget.__init__(self, *args, **kargs)


UnitTemplateWidget.__init__.__doc__ = UnitWaveformsWidget.__init__.__doc__

plot_unit_templates = define_widget_function_from_class(UnitTemplateWidget, 'plot_unit_templates')




