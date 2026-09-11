import numpy as np
import pytest

from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite

from spikeinterface.postprocessing import ComputeAmplitudeScalings
from spikeinterface.postprocessing.amplitude_scalings import _ordinary_scaling_slope, fit_collision


def test_ordinary_scaling_slope_float32_precision():
    """
    The closed-form slope must not lose precision for float32 template/waveform
    inputs relative to an independent float64 `linregress` oracle. At amplitude-scale
    magnitudes (raw ADC/uV range), accumulating in float32 instead of float64 produces
    an error orders of magnitude above float64 rounding noise.
    """
    from scipy.stats import linregress

    rng = np.random.default_rng(2205)
    template = rng.normal(scale=300, size=90).astype(np.float32)
    local_waveform = (1.4 * template + rng.normal(scale=20, size=90).astype(np.float32)).astype(np.float32)

    slope = _ordinary_scaling_slope(template.copy(), local_waveform.copy())
    oracle = linregress(template.astype(np.float64), local_waveform.astype(np.float64)).slope

    assert abs(slope - oracle) < 1e-9


def test_fit_collision_recovers_positive_coefficients():
    """
    `fit_collision` must recover the known, positive scaling factors of two temporally
    overlapping spikes, and must be insensitive to a constant offset added to the traces
    (this exercises the centered non-negative least-squares fit: `positive=True` plus
    `fit_intercept=True` reproduced by centering before `scipy.optimize.nnls`).
    """
    cut_out_before, cut_out_after = 5, 10
    nbefore = cut_out_before
    template_length = nbefore + cut_out_after

    rng = np.random.default_rng(7)
    template_0 = rng.normal(scale=200, size=template_length).astype(np.float32)
    template_1 = rng.normal(scale=200, size=template_length).astype(np.float32)
    all_templates = np.stack([template_0, template_1])[:, :, np.newaxis]
    sparsity_mask = np.ones((2, 1), dtype=bool)

    true_scalings = np.array([1.3, 0.7])
    spike_0_sample, spike_1_sample = 20, 23  # 3 samples apart: their cut-out windows overlap

    traces = np.zeros((50, 1), dtype=np.float32)
    traces[spike_0_sample - cut_out_before : spike_0_sample + cut_out_after, 0] += true_scalings[0] * template_0
    traces[spike_1_sample - cut_out_before : spike_1_sample + cut_out_after, 0] += true_scalings[1] * template_1
    traces += 50.0  # constant offset: must not bias the fit if centering is correct
    traces += rng.normal(scale=0.5, size=traces.shape).astype(np.float32)  # small noise

    collision = np.array(
        [(spike_0_sample, 0), (spike_1_sample, 1)],
        dtype=[("sample_index", "int64"), ("unit_index", "int64")],
    )

    recovered = fit_collision(collision, traces, nbefore, all_templates, sparsity_mask, cut_out_before, cut_out_after)

    assert np.all(recovered >= 0)  # positive=True
    np.testing.assert_allclose(recovered, true_scalings, atol=0.05)


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
