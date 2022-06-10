from spikeinterface.core import waveform_extractor
import spikeinterface.widgets as sw
import spikeinterface.extractors as se
import spikeinterface as si
import kachery_cloud as kcl

with kcl.TemporaryDirectory() as tmpdir:
    R, S = se.toy_example(seed=0)
    R: si.BaseRecording = R
    W = si.extract_waveforms(recording=R, sorting=S, folder=tmpdir + '/waveforms')
    W.set_params(dtype='float')
    W.run_extract_waveforms()
    # W.precompute_templates()

    sw.plot_unit_waveforms(
        waveform_extractor=W,
        backend='sortingview'
    )