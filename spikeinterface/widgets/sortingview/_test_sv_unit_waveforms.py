from spikeinterface.core import waveform_extractor
import spikeinterface.widgets as sw
import spikeinterface.extractors as se
import spikeinterface as si

R, S = se.toy_example(seed=0)
R: si.BaseRecording = R
W = si.WaveformExtractor(recording=R, sorting=S, folder='tmp')
W.set_params(dtype='float')
W.run_extract_waveforms()
# W.precompute_templates()

sw.plot_unit_waveforms(
    waveform_extractor=W,
    backend='sortingview',
    max_channels=6
)