import numpy as np
import pytest

torch = pytest.importorskip("torch")
medicine = pytest.importorskip("medicine")

from spikeinterface import get_noise_levels
from spikeinterface.core.node_pipeline import ExtractDenseWaveforms, run_node_pipeline
from spikeinterface.sortingcomponents.motion import estimate_motion
from spikeinterface.sortingcomponents.peak_detection import detect_peak_methods
from spikeinterface.sortingcomponents.peak_localization.method_list import LocalizeCenterOfMass
from spikeinterface.sortingcomponents.tests.common import make_dataset

# Kept tiny so the model trains in a fraction of a second. Not representative of settings
# one would use for real motion estimation.
_FAST_TRAINING_KWARGS = dict(batch_size=32, training_steps=15, motion_noise_steps=5)


@pytest.fixture(scope="module", name="peaks_and_locations")
def peaks_and_locations_fixture():
    recording, _ = make_dataset()
    noise_levels = get_noise_levels(recording, return_in_uV=False)
    peak_detector_class = detect_peak_methods["locally_exclusive"]
    peak_detector = peak_detector_class(
        recording,
        noise_levels=noise_levels,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=1.0,
        return_output=True,
    )
    extract_dense_waveforms = ExtractDenseWaveforms(
        recording, ms_before=0.1, ms_after=0.3, return_output=False, parents=[peak_detector]
    )
    peak_localizer = LocalizeCenterOfMass(
        recording,
        parents=[peak_detector, extract_dense_waveforms],
        radius_um=60.0,
    )
    peaks, peak_locations = run_node_pipeline(
        recording,
        nodes=[peak_detector, extract_dense_waveforms, peak_localizer],
        job_kwargs=dict(chunk_size=10000, progress_bar=False),
    )
    return recording, peaks, peak_locations


@pytest.mark.parametrize(
    "rigid, win_scale_um, expected_num_depth_bins",
    [
        # non-rigid: MEDiCINe's own recommended default of 2 depth bins, regardless of win_scale_um
        # (win_scale_um/win_margin_um are generic windowing parameters used by other motion estimation
        # methods and should have no bearing on MEDiCINe's num_depth_bins).
        (False, 300.0, 2),
        (False, 1000.0, 2),
        # rigid: a single depth bin, same as MEDiCINe's own rigid convention.
        (True, 300.0, 1),
    ],
)
def test_medicine_default_num_depth_bins(peaks_and_locations, rigid, win_scale_um, expected_num_depth_bins):
    """The spikeinterface `medicine` estimate_motion method should resolve num_depth_bins to MEDiCINe's own
    recommended default, independent of the generic non-rigid window parameters used by other methods."""
    recording, peaks, peak_locations = peaks_and_locations

    motion = estimate_motion(
        recording,
        peaks,
        peak_locations,
        direction="y",
        rigid=rigid,
        method="medicine",
        win_scale_um=win_scale_um,
        **_FAST_TRAINING_KWARGS,
    )

    assert motion.displacement[0].shape[1] == expected_num_depth_bins


def test_medicine_wrapper_matches_original_package(peaks_and_locations):
    """Calling the spikeinterface `medicine` estimate_motion method (with its default num_depth_bins) should
    give the exact same outputs as calling the original medicine.run_medicine() with its own default
    num_depth_bins directly, given the same peaks and RNG state."""
    recording, peaks, peak_locations = peaks_and_locations

    peak_times = peaks["sample_index"] / recording.get_sampling_frequency()
    peak_depths = peak_locations["y"]
    peak_amplitudes = peaks["amplitude"]

    # Run the original MEDiCINe package directly, relying on its own default num_depth_bins.
    torch.manual_seed(0)
    np.random.seed(0)
    _, ref_time_bins, ref_depth_bins, ref_pred_motion = medicine.run_medicine(
        peak_times=peak_times,
        peak_depths=peak_depths,
        peak_amplitudes=peak_amplitudes,
        output_dir=None,
        plot_figures=False,
        **_FAST_TRAINING_KWARGS,
    )

    # Run through the spikeinterface wrapper without specifying num_depth_bins, letting it fall back to
    # its default, and check it lands on the exact same settings and outputs.
    torch.manual_seed(0)
    np.random.seed(0)
    motion = estimate_motion(
        recording,
        peaks,
        peak_locations,
        direction="y",
        rigid=False,
        method="medicine",
        **_FAST_TRAINING_KWARGS,
    )

    np.testing.assert_array_equal(motion.temporal_bins_s[0], ref_time_bins)
    np.testing.assert_array_equal(motion.spatial_bins_um, ref_depth_bins)
    np.testing.assert_array_equal(motion.displacement[0], ref_pred_motion)


if __name__ == "__main__":
    fixture = peaks_and_locations_fixture()
    test_medicine_default_num_depth_bins(fixture, False, 300.0, 2)
    test_medicine_default_num_depth_bins(fixture, False, 1000.0, 2)
    test_medicine_default_num_depth_bins(fixture, True, 300.0, 1)
    test_medicine_wrapper_matches_original_package(fixture)
