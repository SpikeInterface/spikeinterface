import os
import numpy as np
from .base_sv import SortingviewPlotter


class UnitWaveformPlotter(SortingviewPlotter):
    def do_plot(self, data_plot):
        try:
            import figurl as fig
        except ModuleNotFoundError:
            raise Exception('Figurl is not installed. See https://github.com/scratchrealm/figurl2')

        d = data_plot
        unit_ids = d['unit_ids']
        channel_ids = d['channel_ids']
        templates = d['templates']
        channel_locations = d['channel_locations']
        channel_inds = d['channel_inds']
        sampling_frequency = d['sampling_frequency']

        plots = []
        for ii, unit_id in enumerate(unit_ids):
            channel_inds0 = channel_inds[unit_id]
            channel_ids0 = [channel_ids[ind] for ind in channel_inds0]
            channel_locations0 = channel_locations[channel_inds0, :]
            template0 = templates[ii, :, :][:, channel_inds0]

            plots.append({
                'unitId': unit_id,
                'channelIds': channel_ids0,
                'waveform': template0.T
            })
        
        channel_locations0 = {}
        for ii, channel_id in enumerate(channel_ids):
            channel_locations0[str(channel_id)] = channel_locations[ii, :].astype(np.float32)
        data = {
            'type': 'AverageWaveforms',
            'averageWaveforms': plots,
            'samplingFrequency': sampling_frequency,
            'noiseLevel': float(1), # todo
            'channelLocations': channel_locations
        }
        
        F = fig.Figure(
            view_url=os.getenv('SPIKESORTINGVIEW_URL', 'gs://figurl/spikesortingview-4'),
            data=data
        )
        url = F.url(label='Average waveforms')
        print(url)
        return url


from ..unit_waveforms import UnitWaveformsWidget
UnitWaveformPlotter.register(UnitWaveformsWidget)

