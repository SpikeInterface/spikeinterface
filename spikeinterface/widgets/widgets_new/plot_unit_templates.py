from .plot_unit_waveforms import plot_unit_waveforms


def plot_unit_templates(
    *args, **kwargs
):
    kwargs['plot_waveforms'] = False
    return plot_unit_waveforms(*args, **kwargs)