from pathlib import Path

import numpy as np
import pytest
from spikeinterface.core.node_pipeline import ExtractDenseWaveforms
from spikeinterface.sortingcomponents.motion import estimate_motion
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import LocalizeCenterOfMass
from spikeinterface.sortingcomponents.tests.common import make_dataset


DEBUG = False

if DEBUG:
    import matplotlib.pyplot as plt

    plt.ion()
    plt.show()


def setup_dataset_and_peaks(cache_folder):
    print(cache_folder, type(cache_folder))
    cache_folder.mkdir(parents=True, exist_ok=True)

    recording, sorting = make_dataset()
    # detect and localize
    extract_dense_waveforms = ExtractDenseWaveforms(recording, ms_before=0.1, ms_after=0.3, return_output=False)
    pipeline_nodes = [
        extract_dense_waveforms,
        LocalizeCenterOfMass(recording, parents=[extract_dense_waveforms], radius_um=60.0),
    ]
    peaks, peak_locations = detect_peaks(
        recording,
        method="locally_exclusive",
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        chunk_size=10000,
        progress_bar=True,
        pipeline_nodes=pipeline_nodes,
    )

    peaks_path = cache_folder / "dataset_peaks.npy"
    np.save(peaks_path, peaks)
    peak_location_path = cache_folder / "dataset_peak_locations.npy"
    np.save(peak_location_path, peak_locations)

    return recording, sorting, cache_folder


@pytest.fixture(scope="module", name="dataset")
def dataset_fixture(create_cache_folder):
    cache_folder = create_cache_folder / "motion_estimation"
    return setup_dataset_and_peaks(cache_folder)


def test_estimate_motion(dataset):
    # recording, sorting = make_dataset()
    recording, sorting, cache_folder = dataset

    peaks = np.load(cache_folder / "dataset_peaks.npy")
    peak_locations = np.load(cache_folder / "dataset_peak_locations.npy")

    # test many case and sub case
    all_cases = {
        # new york
        "rigid / decentralized / torch": dict(
            rigid=True,
            method="decentralized",
            conv_engine="torch",
            time_horizon_s=None,
        ),
        "rigid / decentralized / numpy": dict(
            rigid=True,
            method="decentralized",
            conv_engine="numpy",
            time_horizon_s=None,
        ),
        "rigid / decentralized / torch / time_horizon_s": dict(
            rigid=True,
            method="decentralized",
            conv_engine="torch",
            time_horizon_s=5,
        ),
        "rigid / decentralized / numpy / time_horizon_s": dict(
            rigid=True,
            method="decentralized",
            conv_engine="numpy",
            time_horizon_s=5,
        ),
        "non-rigid / decentralized / torch": dict(
            rigid=False,
            method="decentralized",
            conv_engine="torch",
            time_horizon_s=None,
        ),
        "non-rigid / decentralized / numpy": dict(
            rigid=False,
            method="decentralized",
            conv_engine="numpy",
            time_horizon_s=None,
        ),
        "non-rigid / decentralized / torch / spatial_prior": dict(
            rigid=False,
            method="decentralized",
            conv_engine="torch",
            time_horizon_s=None,
            spatial_prior=True,
            convergence_method="lsmr",
        ),
        "non-rigid / decentralized / numpy / spatial_prior": dict(
            rigid=False,
            method="decentralized",
            conv_engine="numpy",
            time_horizon_s=None,
            spatial_prior=True,
            convergence_method="lsmr",
        ),
        "non-rigid / decentralized / torch / time_horizon_s": dict(
            rigid=False,
            method="decentralized",
            conv_engine="torch",
            time_horizon_s=5.0,
        ),
        "non-rigid / decentralized / numpy / time_horizon_s": dict(
            rigid=False,
            method="decentralized",
            conv_engine="numpy",
            time_horizon_s=5.0,
        ),
        "non-rigid / decentralized / torch / gradient_descent": dict(
            rigid=False,
            method="decentralized",
            conv_engine="torch",
            convergence_method="gradient_descent",
            time_horizon_s=None,
        ),
        # kilosort 2.5
        "rigid / iterative_template": dict(
            method="iterative_template",
            rigid=True,
        ),
        "non-rigid / iterative_template": dict(
            method="iterative_template",
            rigid=False,
        ),
    }

    motions = {}
    for name, cases_kwargs in all_cases.items():
        print(name)

        kwargs = dict(
            direction="y",
            bin_s=1.0,
            bin_um=10.0,
            margin_um=5,
            extra_outputs=True,
        )
        kwargs.update(cases_kwargs)

        motion, extra = estimate_motion(recording, peaks, peak_locations, **kwargs)
        motions[name] = motion

        if cases_kwargs["rigid"]:
            assert motion.displacement[0].shape[1] == 1
        else:
            assert motion.displacement[0].shape[1] > 1

        # # Test saving to disk
        # corrected_rec = InterpolateMotionRecording(
        #     recording, motion, temporal_bins, spatial_bins, border_mode="force_extrapolate"
        # )
        # rec_folder = cache_folder / (name.replace("/", "").replace(" ", "_") + "_recording")
        # if rec_folder.exists():
        #     shutil.rmtree(rec_folder)
        # corrected_rec.save(folder=rec_folder)

        if DEBUG:
            fig, ax = plt.subplots()
            seg_index = 0
            ax.plot(motion.temporal_bins_s[0], motion.displacement[seg_index])

            # motion_histogram = extra_check['motion_histogram']
            # spatial_hist_bins = extra_check['spatial_hist_bin_edges']
            # fig, ax = plt.subplots()
            # extent = (temporal_bins[0], temporal_bins[-1], spatial_hist_bins[0], spatial_hist_bins[-1])
            # im = ax.imshow(motion_histogram.T, interpolation='nearest',
            #                     origin='lower', aspect='auto', extent=extent)

            # fig, ax = plt.subplots()
            # pairwise_displacement = extra_check['pairwise_displacement_list'][0]
            # im = ax.imshow(pairwise_displacement, interpolation='nearest',
            #                     cmap='PiYG', origin='lower', aspect='auto', extent=None)
            # im.set_clim(-40, 40)
            # ax.set_aspect('equal')
            # fig.colorbar(im)

            plt.show()

    # same params with differents engine should be the same
    motion0 = motions["rigid / decentralized / torch"]
    motion1 = motions["rigid / decentralized / numpy"]
    assert motion0 == motion1

    motion0 = motions["rigid / decentralized / torch / time_horizon_s"]
    motion1 = motions["rigid / decentralized / numpy / time_horizon_s"]
    np.testing.assert_array_almost_equal(motion0.displacement, motion1.displacement)

    motion0 = motions["non-rigid / decentralized / torch"]
    motion1 = motions["non-rigid / decentralized / numpy"]
    np.testing.assert_array_almost_equal(motion0.displacement, motion1.displacement)

    motion0 = motions["non-rigid / decentralized / torch / time_horizon_s"]
    motion1 = motions["non-rigid / decentralized / numpy / time_horizon_s"]
    np.testing.assert_array_almost_equal(motion0.displacement, motion1.displacement)

    motion0 = motions["non-rigid / decentralized / torch / spatial_prior"]
    motion1 = motions["non-rigid / decentralized / numpy / spatial_prior"]
    np.testing.assert_array_almost_equal(motion0.displacement, motion1.displacement)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdirname:
        cache_folder = Path(tmpdirname)
    args = setup_dataset_and_peaks(cache_folder)
    test_estimate_motion(args)
