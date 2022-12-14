# This is kept in 0.97.0 and then will be removed

import warnings

import spikeinterface.core.template_tools as tt


def _warn():
    warnings.warn("The spikeinterface.postprocessing.template_tools is submodule is deprecated."
                  "Use spikeinterface.core.template_tools instead",
                  DeprecationWarning, stacklevel=2)


def get_template_amplitudes(*args, **kwargs):
    _warn()
    return tt.get_template_amplitudes(*args, **kwargs)

def get_template_extremum_channel(*args, **kwargs):
    _warn()
    return tt.get_template_extremum_channel(*args, **kwargs)


def get_template_channel_sparsity(*args, **kwargs):
    _warn()
    return tt.get_template_channel_sparsity(*args, **kwargs)

def get_template_extremum_channel_peak_shift(*args, **kwargs):
    _warn()
    return tt.get_template_extremum_channel_peak_shift(*args, **kwargs)

def get_template_extremum_amplitude(*args, **kwargs):
    _warn()
    return tt.get_template_extremum_amplitude(*args, **kwargs)
