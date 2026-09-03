import numpy as np
import pytest

from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite

from spikeinterface.postprocessing import ComputeAmplitudeScalings
from spikeinterface.postprocessing.amplitude_scalings import fit_collision, _scale_local_waveform_to_uV


class TestAmplitudeScalingsExtension(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize("params", [dict(handle_collisions=True), dict(handle_collisions=False)])
    def test_extension(self, params):
        self.run_extension_tests(ComputeAmplitudeScalings, params)

    def test_scaling_values(self):
        """
        Amplitude finds the scaling factor for each waveform
        to best match its unit template. In this test, amplitude scalings
        are calculated from the `sorting_analyzer`. In the test environment,
        injected waveforms are not scaled from the template and so
        should only differ by Gaussian noise. Therefore the median
        scaling should be close to 1.
        """
        sorting_analyzer = self._prepare_sorting_analyzer(
            "memory", sparse=True, extension_class=ComputeAmplitudeScalings
        )
        sorting_analyzer.compute("amplitude_scalings", handle_collisions=False)

        spikes = sorting_analyzer.sorting.to_spike_vector()

        ext = sorting_analyzer.get_extension("amplitude_scalings")

        for unit_index, unit_id in enumerate(sorting_analyzer.unit_ids):
            mask = spikes["unit_index"] == unit_index
            scalings = ext.data["amplitude_scalings"][mask]
            median_scaling = np.median(scalings)
            np.testing.assert_array_equal(np.round(median_scaling), 1)


@pytest.mark.parametrize(
    "sparse_indices", [np.array([0, 1, 2]), np.array([2, 3, 4]), np.array([5, 6, 7]), np.array([0, 3, 7])]
)
def test_scale_local_waveform_to_uv_uses_its_own_channels_gain(sparse_indices):
    """
    `_scale_local_waveform_to_uV` is the scaling step shared by the two places that cut a
    window out of the traces: the per-spike loop in `AmplitudeScalingNode.compute` and
    `fit_collision`. It has to pick each channel's own gain/offset out of the chunk-wide
    `gains`/`offsets` arrays, not some other subset. Scaling the cut-out window must equal
    cutting the same window out of traces that were already scaled beforehand, for whichever
    channel subset the caller cut out.
    """
    rng = np.random.default_rng(seed=2205)
    num_channels, num_samples = 8, 50
    gains = rng.uniform(0.5, 4.0, size=num_channels)
    offsets = rng.uniform(-20.0, 20.0, size=num_channels)
    raw_traces = rng.integers(-2000, 2000, size=(num_samples, num_channels)).astype("int16")

    local_waveform = raw_traces[10:30, sparse_indices]
    scaled_inside = _scale_local_waveform_to_uV(local_waveform, gains, offsets, sparse_indices)

    scaled_beforehand = (raw_traces.astype("float32") * gains + offsets)[10:30, sparse_indices]

    np.testing.assert_array_equal(scaled_inside, scaled_beforehand)


def test_fit_collision_scales_each_channel_with_its_own_gain():
    """
    `fit_collision` scales the traces it cuts out to uV itself, so it has to apply
    the gain of each channel to that channel. Scaling inside must give exactly what
    passing already scaled traces gives. A gain taken from the wrong channel still
    returns a well formed scaling, so the two paths are compared directly, with a
    gain and an offset which are different on every channel.
    """
    rng = np.random.default_rng(seed=2205)
    num_channels, num_units = 8, 3
    nbefore, nafter, num_samples = 20, 30, 400

    raw_traces = rng.integers(-2000, 2000, size=(num_samples, num_channels)).astype("int16")
    all_templates = (rng.normal(size=(num_units, nbefore + nafter, num_channels)) * 50).astype("float32")
    # unit 0 and unit 1 (the colliding pair below) are on non-contiguous, interleaved
    # channels so their union isn't a single slice: sparse_indices must stay [0, 1, 3, 4, 6, 7].
    sparsity_mask = np.zeros((num_units, num_channels), dtype=bool)
    sparsity_mask[0, [0, 3, 6]] = True
    sparsity_mask[1, [1, 4, 7]] = True
    sparsity_mask[2, [2, 5]] = True

    gains = rng.uniform(0.5, 4.0, size=num_channels)
    offsets = rng.uniform(-20.0, 20.0, size=num_channels)

    collision = np.zeros(2, dtype=[("sample_index", "int64"), ("unit_index", "int64")])
    collision["sample_index"] = [180, 190]
    collision["unit_index"] = [0, 1]

    scaled_inside = fit_collision(
        collision, raw_traces, nbefore, all_templates, sparsity_mask, nbefore, nafter, gains, offsets
    )
    scaled_traces = raw_traces.astype("float32") * gains + offsets
    scaled_beforehand = fit_collision(collision, scaled_traces, nbefore, all_templates, sparsity_mask, nbefore, nafter)

    np.testing.assert_array_equal(scaled_inside, scaled_beforehand)
