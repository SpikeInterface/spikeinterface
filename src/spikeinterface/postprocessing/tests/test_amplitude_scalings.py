import numpy as np
import pytest

from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite

from spikeinterface.postprocessing import ComputeAmplitudeScalings
from spikeinterface.core.base import spike_peak_dtype
from spikeinterface.postprocessing.amplitude_scalings import (
    HAVE_NUMBA,
    AmplitudeScalingNode,
    _channel_lists,
    _ordinary_scaling_slope,
    _unit_pair_overlap_matrix,
    find_collisions,
    fit_collision,
)


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


def test_find_collisions_with_margin_indices(monkeypatch):
    dtype = spike_peak_dtype + [("in_margin", "bool")]
    spikes_within_margin = np.array(
        [
            (8, 0, -1.0, 0, 1, True),
            (10, 0, -1.0, 0, 0, False),
            (10, 1, -1.0, 0, 1, False),
            (13, 2, -1.0, 0, 2, False),
            (30, 1, -1.0, 0, 1, True),
        ],
        dtype=dtype,
    )
    spike_indices = np.flatnonzero(~spikes_within_margin["in_margin"])
    spikes = spikes_within_margin[spike_indices]
    sparsity_mask = np.array([[True, False], [True, True], [False, True]])

    with monkeypatch.context() as patch_context:
        patch_context.setattr(np, "where", lambda *_: pytest.fail("spike indices were searched again"))
        collisions = find_collisions(
            spikes,
            spikes_within_margin,
            delta_collision_samples=4,
            overlap_matrix=_unit_pair_overlap_matrix(sparsity_mask),
            spike_indices=spike_indices,
        )

    assert set(collisions) == {0, 1, 2}
    np.testing.assert_array_equal(collisions[0], spikes_within_margin[[1, 0, 2]])
    np.testing.assert_array_equal(collisions[1], spikes_within_margin[[2, 0, 1, 3]])
    np.testing.assert_array_equal(collisions[2], spikes_within_margin[[3, 2]])


def test_unit_pair_overlap_matrix_matches_naive_reference():
    """
    `_unit_pair_overlap_matrix` replaces a per-pair `np.any(sparsity_mask[i] & sparsity_mask[j])`
    lookup (previously recomputed on every temporally-overlapping spike-pair candidate inside
    `find_collisions`, millions of times on a realistic recording) with one matrix multiplication
    computed once. Verify it agrees, entry by entry, with the direct naive reference on
    NON-CONTIGUOUS per-unit channel subsets (a bounded/contiguous slice would pass even a mutant
    that only checks a channel range), and on a unit with an entirely empty sparsity row (no
    channels at all -- that unit must not spuriously overlap with anything, including itself).
    """
    rng = np.random.default_rng(99)
    num_units, num_channels = 15, 24

    sparsity_mask = np.zeros((num_units, num_channels), dtype=bool)
    for unit_index in range(num_units):
        if unit_index == 0:
            continue  # unit 0: entirely empty sparsity row
        num_active = rng.integers(1, num_channels // 2)
        active_channels = rng.choice(num_channels, size=num_active, replace=False)  # non-contiguous
        sparsity_mask[unit_index, active_channels] = True

    overlap_matrix = _unit_pair_overlap_matrix(sparsity_mask)

    assert overlap_matrix.shape == (num_units, num_units)
    assert overlap_matrix.dtype == np.bool_
    assert not overlap_matrix[0, :].any()  # the empty-row unit overlaps with nothing
    assert not overlap_matrix[:, 0].any()

    for i in range(num_units):
        for j in range(num_units):
            expected = np.any(sparsity_mask[i] & sparsity_mask[j])
            assert overlap_matrix[i, j] == expected, f"mismatch at unit pair ({i}, {j})"


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


def _scalings_numpy(traces, peaks, templates, sparsity_mask, nbefore, cut_out_before, cut_out_after, delta, gains):
    """The numpy path of `AmplitudeScalingNode.compute`, spike by spike."""
    local_spike_indices = np.flatnonzero(~peaks["in_margin"])
    local_spikes = peaks[local_spike_indices]
    offsets = None if gains is None else np.zeros_like(gains)
    collisions = {}
    if delta >= 0:
        overlap_matrix = _unit_pair_overlap_matrix(sparsity_mask)
        collisions = find_collisions(local_spikes, peaks, delta, overlap_matrix, local_spike_indices)
    scalings = np.zeros(len(local_spikes))
    for spike_index, spike in enumerate(local_spikes):
        if spike_index in collisions:
            collision = collisions[spike_index]
            fit = fit_collision(
                collision, traces, nbefore, templates, sparsity_mask, cut_out_before, cut_out_after, gains, offsets
            )
            scalings[spike_index] = fit[0]
            continue
        unit_index = spike["unit_index"]
        if np.all(templates[unit_index] == 0):
            continue
        (channels,) = np.nonzero(sparsity_mask[unit_index])
        start = max(0, spike["sample_index"] - cut_out_before)
        stop = min(traces.shape[0], spike["sample_index"] + cut_out_after)
        first = nbefore - (spike["sample_index"] - start)
        template = templates[unit_index][first : first + stop - start, channels]
        waveform = traces[start:stop, channels]
        if gains is not None:
            waveform = waveform.astype("float32") * gains[channels] + offsets[channels]
        scalings[spike_index] = _ordinary_scaling_slope(template, waveform)
    collision_mask = np.isin(np.arange(len(local_spikes)), list(collisions))
    return scalings, collision_mask


def _scalings_numba(traces, peaks, templates, sparsity_mask, nbefore, cut_out_before, cut_out_after, delta, gains):
    """The node's numba path, including the spikes it hands back to the numpy path."""
    node = object.__new__(AmplitudeScalingNode)
    node._gains = gains
    node._offsets = None if gains is None else np.zeros_like(gains)
    node._dtype = np.float64  # keep full precision for the comparison
    node._all_templates = templates
    node._sparsity_mask = sparsity_mask
    node._overlap_matrix = _unit_pair_overlap_matrix(sparsity_mask)
    node._nbefore = nbefore
    node._cut_out_before = cut_out_before
    node._cut_out_after = cut_out_after
    node._handle_collisions = delta >= 0
    node._delta_collision_samples = delta
    node._template_is_zero = np.all(templates == 0, axis=(1, 2))
    node._cut_templates = np.ascontiguousarray(templates[:, nbefore - cut_out_before : nbefore + cut_out_after])
    node._sparse_channels = _channel_lists(sparsity_mask)
    return node._compute_numba(traces, peaks)


def _make_chunk(seed, num_units=12, num_channels=32, num_samples=6000, num_spikes=600):
    """Spikes with random, possibly negative amplitudes on non-contiguous channels, including at both edges."""
    rng = np.random.default_rng(seed)
    nbefore, nafter = 20, 30
    sparsity_mask = np.zeros((num_units, num_channels), dtype=bool)
    for unit_index in range(num_units):
        sparsity_mask[unit_index, rng.choice(num_channels, size=8, replace=False)] = True
    templates = (
        rng.normal(scale=50, size=(num_units, nbefore + nafter, num_channels)) * sparsity_mask[:, None]
    ).astype(np.float32)
    templates[-1] = 0  # a unit whose template is zero
    samples = np.sort(
        np.concatenate([rng.integers(0, num_samples, num_spikes - 4), [0, 5, num_samples - 5, num_samples - 1]])
    )
    peaks = np.zeros(num_spikes, dtype=spike_peak_dtype + [("in_margin", "bool")])
    peaks["sample_index"] = samples
    peaks["unit_index"] = rng.integers(0, num_units, num_spikes)
    peaks["in_margin"] = (samples < 100) & (rng.random(num_spikes) < 0.5)
    traces = rng.normal(scale=5, size=(num_samples + nafter, num_channels)).astype(np.float32)
    for sample, unit_index in zip(samples, peaks["unit_index"]):
        start = sample - nbefore
        lo = max(0, -start)
        traces[start + lo : sample + nafter] += rng.uniform(-0.5, 1.5) * templates[unit_index][lo:]
    traces = traces[:num_samples]
    return traces, peaks, templates, sparsity_mask, nbefore


@pytest.mark.skipif(not HAVE_NUMBA, reason="the numba kernel needs numba")
@pytest.mark.parametrize("gains", [None, "float32", "float64"])
@pytest.mark.parametrize("delta", [10, -1])
@pytest.mark.parametrize("cut_out", [(20, 30), (12, 25)])
def test_numba_scalings_match_numpy(gains, delta, cut_out):
    """
    The kernel equals the numpy path up to rounding: with and without gains and collisions, a cut-out shorter
    than the templates, spikes clipped at either edge of the
    traces, binding non-negativity constraints and a zero template.
    """
    traces, peaks, templates, sparsity_mask, nbefore = _make_chunk(0)
    if gains is not None:
        gains = np.linspace(0.1, 0.3, traces.shape[1]).astype(gains)
    args = (traces, peaks, templates, sparsity_mask, nbefore, *cut_out, delta, gains)
    expected, expected_collisions = _scalings_numpy(*args)
    scalings, collisions = _scalings_numba(*args)

    np.testing.assert_array_equal(collisions, expected_collisions)
    assert (delta < 0) or (expected_collisions.mean() > 0.3 and np.sum(expected[expected_collisions] == 0) > 10)
    np.testing.assert_allclose(scalings, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.skipif(not HAVE_NUMBA, reason="the numba kernel needs numba")
@pytest.mark.parametrize("return_in_uV", [True, False])
@pytest.mark.parametrize("sparser", [False, True])
def test_amplitude_scalings_numba_matches_numpy_extension(monkeypatch, return_in_uV, sparser):
    """A sparsity sparser than the templates' sends the node to the numpy path, which must give the same result."""
    from spikeinterface.core import ChannelSparsity, create_sorting_analyzer, generate_ground_truth_recording
    from spikeinterface.postprocessing import amplitude_scalings

    recording, sorting = generate_ground_truth_recording(durations=[10.0], num_units=20, seed=0)
    analyzer = create_sorting_analyzer(sorting, recording, sparse=True, return_in_uV=return_in_uV)
    analyzer.compute(["random_spikes", "templates"])
    params = {}
    if sparser:
        # one channel fewer per unit, keeping its main channel
        mask = analyzer.sparsity.mask.copy()
        main_channels = analyzer.get_main_channels(outputs="index")
        for unit_index, channels in enumerate(mask):
            mask[unit_index, np.flatnonzero(channels & (np.arange(mask.shape[1]) != main_channels[unit_index]))[0]] = (
                False
            )
        params["sparsity"] = ChannelSparsity(mask, analyzer.unit_ids, analyzer.channel_ids)
    numba_chunks = []
    compute_numba = amplitude_scalings.AmplitudeScalingNode._compute_numba
    monkeypatch.setattr(
        amplitude_scalings.AmplitudeScalingNode,
        "_compute_numba",
        lambda self, *args: numba_chunks.append(1) or compute_numba(self, *args),
    )
    results = {}
    for have_numba in (True, False):
        monkeypatch.setattr(amplitude_scalings, "HAVE_NUMBA", have_numba)
        ext = analyzer.compute("amplitude_scalings", chunk_duration="1s", n_jobs=1, **params)
        results[have_numba] = ext.data["amplitude_scalings"], ext.data["collision_mask"]
    assert (len(numba_chunks) == 0) == sparser
    assert results[False][1].any()
    np.testing.assert_array_equal(results[True][1], results[False][1])
    np.testing.assert_allclose(results[True][0], results[False][0], rtol=1e-5)


@pytest.mark.skipif(not HAVE_NUMBA, reason="the numba kernel needs numba")
@pytest.mark.parametrize("lag", [0, 1, 2])
def test_numba_scalings_match_numpy_for_duplicates(lag):
    """
    Two spikes of one unit `lag` samples apart give nearly or exactly equal columns. At lag 0 the fit is singular,
    so the kernel hands it to `fit_collision`, which gives the whole amplitude to the spike being scaled.
    """
    nbefore, nafter, num_channels = 20, 30, 8
    rng = np.random.default_rng(0)
    times = np.arange(nbefore + nafter)
    template = -100 * np.exp(-0.5 * ((times - nbefore) / 2.0) ** 2)[:, None] * rng.uniform(0.3, 1, num_channels)
    templates = template[np.newaxis].astype(np.float32)
    sparsity_mask = np.ones((1, num_channels), dtype=bool)
    peaks = np.zeros(2, dtype=spike_peak_dtype + [("in_margin", "bool")])
    peaks["sample_index"] = [80, 80 + lag]
    traces = rng.normal(size=(200, num_channels)).astype(np.float32)
    for sample, amplitude in zip(peaks["sample_index"], [1.0, 0.6]):
        traces[sample - nbefore : sample + nafter] += amplitude * templates[0]
    args = (traces, peaks, templates, sparsity_mask, nbefore, nbefore, nafter, 10, None)
    expected, expected_collisions = _scalings_numpy(*args)
    scalings, collisions = _scalings_numba(*args)

    assert collisions.all() and expected_collisions.all()
    np.testing.assert_allclose(scalings, expected, rtol=1e-9)
    if lag == 0:
        np.testing.assert_allclose(scalings, 1.6, rtol=0.01)


def _two_unit_collision(second_template_noise, lag=3, trace_noise=3.0, num_channels=8, dtype=np.float32):
    """Two units colliding `lag` samples apart, the second's template the first's plus noise of the given scale."""
    nbefore, nafter = 20, 30
    rng = np.random.default_rng(1)
    times = np.arange(nbefore + nafter)
    first = -100 * np.exp(-0.5 * ((times - nbefore) / 2.0) ** 2)[:, None] * rng.uniform(0.3, 1, num_channels)
    second = first + second_template_noise * rng.normal(size=first.shape)
    templates = np.stack([first, second]).astype(np.float32)
    sparsity_mask = np.ones((2, num_channels), dtype=bool)
    peaks = np.zeros(2, dtype=spike_peak_dtype + [("in_margin", "bool")])
    peaks["sample_index"] = [80, 80 + lag]
    peaks["unit_index"] = [0, 1]
    traces = trace_noise * rng.normal(size=(200, num_channels))
    for sample, unit_index, amplitude in zip(peaks["sample_index"], [0, 1], [1.3, 0.7]):
        traces[sample - nbefore : sample + nafter] += amplitude * templates[unit_index]
    return (traces.astype(dtype), peaks, templates, sparsity_mask, nbefore, nbefore, nafter, 10, None)


@pytest.mark.skipif(not HAVE_NUMBA, reason="the numba kernel needs numba")
@pytest.mark.parametrize("second_template_noise", [1e-5, 1e-3])
def test_numba_scalings_match_numpy_for_nearly_identical_templates(second_template_noise):
    """An ill-conditioned fit goes to `fit_collision`, so it equals the numpy path rather than a truncated fit."""
    args = _two_unit_collision(second_template_noise, lag=0, trace_noise=0.0)
    expected, _ = _scalings_numpy(*args)
    scalings, _ = _scalings_numba(*args)
    assert 0 < expected[0] < 2
    np.testing.assert_allclose(scalings, expected, rtol=1e-9)


@pytest.mark.skipif(not HAVE_NUMBA, reason="the numba kernel needs numba")
def test_numba_scalings_raise_on_nan_in_collision_like_numpy():
    traces, *rest = _two_unit_collision(second_template_noise=10.0)
    traces[82, 0] = np.nan
    for scalings_function in (_scalings_numpy, _scalings_numba):
        with pytest.raises(ValueError):
            scalings_function(traces, *rest)
