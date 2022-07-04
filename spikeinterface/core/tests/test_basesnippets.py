"""
test for BaseSnippets are done with NumpySnippetsExtractor.
but check only for BaseRecording general methods.
"""
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_raises

#from probeinterface import Probe
from spikeinterface.core.testing_tools import generate_snippets
from spikeinterface.core.numpyextractors import NumpySnippetsExtractor

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"
    cache_folder.mkdir(exist_ok=True, parents=True)

def test_BaseSnippets():
    duration = [4, 3]
    num_channels = 3
    nbefore = 20
    nafter = 44
    wf_folder = cache_folder / "wfs"
    wf_folder.mkdir(parents=True)
    nse, sorting = generate_snippets(durations=duration, num_channels=num_channels,
                                        nbefore=nbefore, nafter=nafter, 
                                        wf_folder=wf_folder)

    assert nse.get_probe() is not None
    assert nse.get_num_segments() == len(duration)

    assert np.all(nse.ids_to_indices([0, 1, 2]) == [0, 1, 2])
    assert np.all(nse.ids_to_indices(
        [0, 1, 2], prefer_slice=True) == slice(0, 3, None))

    # annotations / properties
    nse.annotate(gre='ta')
    assert nse.get_annotation('gre') == 'ta'

    nse.set_channel_groups([0, 0, 1])
    groups = nse.get_channel_groups()
    assert np.array_equal(groups, [0, 0, 1])

    nse.set_property('quality', [1., 3.3, np.nan])
    values = nse.get_property('quality')
    assert np.all(values[:2] == [1., 3.3, ])

    # missing property
    nse.set_property('string_property', ["ciao", "bello"], ids=[0, 1])
    values = nse.get_property('string_property')
    assert values[2] == ""

    # setting an different type raises an error
    assert_raises(Exception, nse.set_property, key='string_property_nan', values=["ciao", "bello"], ids=[0, 1],
                  missing_value=np.nan)

    # int properties without missing values raise an error
    assert_raises(Exception, nse.set_property,
                  key='int_property', values=[5, 6], ids=[1, 2])

    nse.set_property('int_property', [5, 6], ids=[1, 2], missing_value=200)
    values = nse.get_property('int_property')
    assert values.dtype.kind == "i"

    times0 = nse.get_frames(segment_index=0)

    seg0_times = sorting.get_all_spike_trains()[0][0]

    assert all(seg0_times==times0)
    # TODO: after making another extractor
    # dump/load dict
    # dump/load json
    # dump/load pickle
    # cache to binary
    #load from folder
    # cache to memory
    # cache joblib several jobs
    # set/get Probe


    # test return_scale
    # nse_int16 = NumpySnippetsExtractor(wfs.astype('int16'), peaks2['sample_ind'], recording.get_sampling_frequency(), nbefore=nbefore, channel_ids=None)
    # assert nse_int16.get_dtype() == 'int16'

    # wfs_int16 = nse_int16.get_snippets()
    # assert wfs_int16.dtype == 'int16'
    # # return_scaled raise error when no gain_to_uV/offset_to_uV properties
    # with pytest.raises(ValueError):
    #     wfs_float32 = nse_int16.get_snippets(return_scaled=True)
    # nse_int16.set_property('gain_to_uV', [.195] * num_channels)
    # nse_int16.set_property('offset_to_uV', [0.] * num_channels)
    # wfs_float32 = nse_int16.get_snippets(return_scaled=True)
    # assert wfs_float32.dtype == 'float32'

    # wfs_from_time = nse_int16.get_snippets_from_frames(start_frame=peaks2['sample_ind'][0], end_frame= peaks2['sample_ind'][5])
    # assert wfs_from_time.shape[0]==5
    

if __name__ == '__main__':
    test_BaseSnippets()
