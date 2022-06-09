

from .base_sv import SortingviewPlotter


class UnitWaveformPlotter(SortingviewPlotter):
    def do_plot(self, data_plot):
        print('SV plotter for UnitWaveform', data_plot.keys())
        
        myfigurl  = 'https://figurl.org/f?v=gs://figurl/glance-raw-traces-1&d=ipfs://QmYDC6aw1dD3NLyvMjzhoZgXaU7XNMRScQ8NLLGS2gacM9&label=Mandelbrot%20tiled%20image'
        print(myfigurl)
        return myfigurl


from ..unitwaveforms import UnitWaveformsWidget
UnitWaveformPlotter.register(UnitWaveformsWidget)

