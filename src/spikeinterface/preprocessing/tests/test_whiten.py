import pytest
import numpy as np
from pathlib import Path

from spikeinterface import set_global_tmp_folder
from spikeinterface.core import generate_recording

from spikeinterface.preprocessing import whiten, scale

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "preprocessing"
else:
    cache_folder = Path("cache_folder") / "preprocessing"

set_global_tmp_folder(cache_folder)


def test_whiten():
    rec = generate_recording()

    rec2 = whiten(rec)
    rec2.save(verbose=False)

    # test dtype
    rec_int = scale(rec2, dtype="int16")
    rec3 = whiten(rec_int, dtype="float16")
    rec3 = rec3.save(folder=cache_folder / "rec1")
    assert rec3.get_dtype() == "float16"

    # test parallel
    rec_par = rec3.save(folder=cache_folder / "rec_par", n_jobs=2)
    np.testing.assert_array_equal(rec3.get_traces(segment_index=0),
                                  rec_par.get_traces(segment_index=0))


if __name__ == '__main__':
    test_whiten()
