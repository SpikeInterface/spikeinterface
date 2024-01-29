import pytest

import numpy as np
import json

from spikeinterface.core import ChannelSparsity, estimate_sparsity
from spikeinterface.core.core_tools import check_json
from spikeinterface.core import generate_ground_truth_recording


def test_ChannelSparsity():
    for unit_ids in (["a", "b", "c", "d"], [4, 5, 6, 7]):
        channel_ids = [1, 2, 3]
        mask = np.zeros((4, 3), dtype="bool")
        mask[
            0,
            [
                0,
            ],
        ] = True
        mask[1, [0, 1, 2]] = True
        mask[2, [0, 2]] = True
        mask[
            3,
            [
                0,
            ],
        ] = True

        sparsity = ChannelSparsity(mask, unit_ids, channel_ids)
        print(sparsity)

        with pytest.raises(AssertionError):
            sparsity = ChannelSparsity(mask, unit_ids, channel_ids[:2])

        for key, v in sparsity.unit_id_to_channel_ids.items():
            assert key in unit_ids
            assert np.all(np.isin(v, channel_ids))

        for key, v in sparsity.unit_id_to_channel_indices.items():
            assert key in unit_ids
            assert np.all(v < len(channel_ids))

        sparsity2 = ChannelSparsity.from_unit_id_to_channel_ids(sparsity.unit_id_to_channel_ids, unit_ids, channel_ids)
        # print(sparsity2)
        assert np.array_equal(sparsity.mask, sparsity2.mask)

        d = sparsity.to_dict()
        # print(d)
        sparsity3 = ChannelSparsity.from_dict(d)
        assert np.array_equal(sparsity.mask, sparsity3.mask)
        # print(sparsity3)

        d2 = json.loads(json.dumps(check_json(d)))
        sparsity4 = ChannelSparsity.from_dict(d2)
        assert np.array_equal(sparsity.mask, sparsity4.mask)


def test_sparsify_waveforms():
    seed = 0
    rng = np.random.default_rng(seed=seed)

    num_units = 3
    num_samples = 5
    num_channels = 4

    is_mask_valid = False
    while not is_mask_valid:
        sparsity_mask = rng.integers(0, 1, size=(num_units, num_channels), endpoint=True, dtype="bool")
        is_mask_valid = np.all(sparsity_mask.sum(axis=1) > 0)

    unit_ids = np.arange(num_units)
    channel_ids = np.arange(num_channels)
    sparsity = ChannelSparsity(mask=sparsity_mask, unit_ids=unit_ids, channel_ids=channel_ids)

    for unit_id in unit_ids:
        waveforms_dense = rng.random(size=(num_units, num_samples, num_channels))

        # Test are_waveforms_dense
        assert sparsity.are_waveforms_dense(waveforms_dense)

        # Test sparsify
        waveforms_sparse = sparsity.sparsify_waveforms(waveforms_dense, unit_id=unit_id)
        non_zero_indices = sparsity.unit_id_to_channel_indices[unit_id]
        num_active_channels = len(non_zero_indices)
        assert waveforms_sparse.shape == (num_units, num_samples, num_active_channels)

        # Test round-trip (note that this is loosy)
        unit_id = unit_ids[unit_id]
        non_zero_indices = sparsity.unit_id_to_channel_indices[unit_id]
        waveforms_dense2 = sparsity.densify_waveforms(waveforms_sparse, unit_id=unit_id)
        assert np.array_equal(waveforms_dense[..., non_zero_indices], waveforms_dense2[..., non_zero_indices])

        # Test sparsify with one waveform (template)
        template_dense = waveforms_dense.mean(axis=0)
        template_sparse = sparsity.sparsify_waveforms(template_dense, unit_id=unit_id)
        assert template_sparse.shape == (num_samples, num_active_channels)

        # Test round trip with template
        template_dense2 = sparsity.densify_waveforms(template_sparse, unit_id=unit_id)
        assert np.array_equal(template_dense[..., non_zero_indices], template_dense2[:, non_zero_indices])


def test_densify_waveforms():
    seed = 0
    rng = np.random.default_rng(seed=seed)

    num_units = 3
    num_samples = 5
    num_channels = 4

    is_mask_valid = False
    while not is_mask_valid:
        sparsity_mask = rng.integers(0, 1, size=(num_units, num_channels), endpoint=True, dtype="bool")
        is_mask_valid = np.all(sparsity_mask.sum(axis=1) > 0)

    unit_ids = np.arange(num_units)
    channel_ids = np.arange(num_channels)
    sparsity = ChannelSparsity(mask=sparsity_mask, unit_ids=unit_ids, channel_ids=channel_ids)

    for unit_id in unit_ids:
        non_zero_indices = sparsity.unit_id_to_channel_indices[unit_id]
        num_active_channels = len(non_zero_indices)
        waveforms_sparse = rng.random(size=(num_units, num_samples, num_active_channels))

        # Test are waveforms sparse
        assert sparsity.are_waveforms_sparse(waveforms_sparse, unit_id=unit_id)

        # Test densify
        waveforms_dense = sparsity.densify_waveforms(waveforms_sparse, unit_id=unit_id)
        assert waveforms_dense.shape == (num_units, num_samples, num_channels)

        # Test round-trip
        waveforms_sparse2 = sparsity.sparsify_waveforms(waveforms_dense, unit_id=unit_id)
        assert np.array_equal(waveforms_sparse, waveforms_sparse2)

        # Test densify with one waveform (template)
        template_sparse = waveforms_sparse.mean(axis=0)
        template_dense = sparsity.densify_waveforms(template_sparse, unit_id=unit_id)
        assert template_dense.shape == (num_samples, num_channels)

        # Test round trip with template
        template_sparse2 = sparsity.sparsify_waveforms(template_dense, unit_id=unit_id)
        assert np.array_equal(template_sparse, template_sparse2)


def test_estimate_sparsity():
    num_units = 5
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0],
        sampling_frequency=16000.0,
        num_channels=10,
        num_units=5,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_level=1.0, strategy="tile_pregenerated"),
        seed=2205,
    )

    # small radius should give a very sparse = one channel per unit
    sparsity = estimate_sparsity(
        recording,
        sorting,
        num_spikes_for_sparsity=50,
        ms_before=1.0,
        ms_after=2.0,
        method="radius",
        radius_um=1.0,
        chunk_duration="1s",
        progress_bar=True,
        n_jobs=2,
    )
    # print(sparsity)
    assert np.array_equal(np.sum(sparsity.mask, axis=1), np.ones(num_units))

    # best_channel : the mask should exactly 3 channels per units
    sparsity = estimate_sparsity(
        recording,
        sorting,
        num_spikes_for_sparsity=50,
        ms_before=1.0,
        ms_after=2.0,
        method="best_channels",
        num_channels=3,
        chunk_duration="1s",
        progress_bar=True,
        n_jobs=1,
    )
    assert np.array_equal(np.sum(sparsity.mask, axis=1), np.ones(num_units) * 3)


if __name__ == "__main__":
    test_ChannelSparsity()
    test_estimate_sparsity()
