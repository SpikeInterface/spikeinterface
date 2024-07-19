import pytest
import numpy as np

from spikeinterface.core import generate_recording

from spikeinterface.preprocessing import whiten, scale, compute_whitening_matrix


def test_whiten(create_cache_folder):
    cache_folder = create_cache_folder
    rec = generate_recording(num_channels=4, seed=2205)

    print(rec.get_channel_locations())
    random_chunk_kwargs = {}
    W1, M = compute_whitening_matrix(rec, "global", random_chunk_kwargs, apply_mean=False, radius_um=None)
    # print(W)
    # print(M)

    with pytest.raises(AssertionError):
        W, M = compute_whitening_matrix(rec, "local", random_chunk_kwargs, apply_mean=False, radius_um=None)
    W, M = compute_whitening_matrix(rec, "local", random_chunk_kwargs, apply_mean=False, radius_um=25)
    # W must be sparse
    np.sum(W == 0) == 6

    rec2 = whiten(rec)
    rec2.save(verbose=False)

    # test dtype
    rec_int = scale(rec2, dtype="int16")
    rec3 = whiten(rec_int, dtype="float16")
    rec3 = rec3.save(folder=cache_folder / "rec1")
    assert rec3.get_dtype() == "float16"

    # test parallel
    rec_par = rec3.save(folder=cache_folder / "rec_par", n_jobs=2)
    np.testing.assert_array_equal(rec3.get_traces(segment_index=0), rec_par.get_traces(segment_index=0))

    with pytest.raises(AssertionError):
        rec4 = whiten(rec_int, dtype=None)
    rec4 = whiten(rec_int, dtype=None, int_scale=256)
    assert rec4.get_dtype() == "int16"
    assert rec4._kwargs["M"] is None

    # test regularization : norm should be smaller
    W2, M = compute_whitening_matrix(rec, "global", random_chunk_kwargs, apply_mean=False, regularize=True)
    assert np.linalg.norm(W1) > np.linalg.norm(W2)


if __name__ == "__main__":
    test_whiten()
