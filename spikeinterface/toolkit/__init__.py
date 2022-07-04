import warnings

warnings.warn("The 'toolkit' module is deprecated. "
              "Use spikeinterface.preprocessing/postptocessing/qualitymetrics instead",
              DeprecationWarning, stacklevel=2)

from . import preprocessing
from .. import postprocessing
from .. import qualitymetrics

from ..core.recording_tools import (get_random_data_chunks, get_channel_distances,
                                    get_closest_channels, get_noise_levels)

# this make function directly accsible with too much nested submodule
#  st.bandpass_filter(...) and also st.preprocessing.bandpass_filter(...)
from ..preprocessing import *
from ..postprocessing import *
from ..qualitymetrics import *
