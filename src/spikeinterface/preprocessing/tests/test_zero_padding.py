import pytest
from pathlib import Path
import numpy as np

from spikeinterface import set_global_tmp_folder
from spikeinterface.core import generate_recording

from spikeinterface.preprocessing import zero_channel_pad

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "preprocessing"
else:
    cache_folder = Path("cache_folder") / "preprocessing"

set_global_tmp_folder(cache_folder)

def test_zero_padding():
    num_original_channels = 4
    num_padded_channels = num_original_channels + 8
    rec = generate_recording(num_channels=num_original_channels, durations=[10])

    rec2 = zero_channel_pad(rec, num_channels=num_padded_channels)
    rec2.save(verbose=False)
    
    print(rec2)
    
    assert rec2.get_num_channels() == num_padded_channels
    
    tr = rec2.get_traces()
    assert np.allclose(tr[:, num_original_channels:], 
                       np.zeros((rec2.get_num_samples(), num_padded_channels - num_original_channels)))
    assert np.allclose(tr[:, :num_original_channels], rec.get_traces())


if __name__ == '__main__':
    test_zero_padding()
