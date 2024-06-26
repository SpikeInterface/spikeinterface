import copy
from typing import Any
import spikeinterface.full as si
import numpy as np
import torch
import kilosort
from kilosort.io import load_probe
import pandas as pd

import pytest
from probeinterface.io import write_prb
from kilosort.parameters import DEFAULT_SETTINGS
from packaging.version import parse
from importlib.metadata import version

# TODO: duplicate_spike_bins to duplicate_spike_ms
# TODO: write an issue on KS about bin! vs bin_ms!
# TODO: expose tmin and tmax
# TODO: expose save_preprocessed_copy
# TODO: make here a log of all API changes (or on kilosort4.py)
# TODO: try out longer recordings and do some benchmarking tests..
# TODO: expose tmin and tmax
# There is no way to skip HP spatial filter
# might as well expose tmin and tmax
# might as well expose preprocessing save (across the two functions that use it)
# BinaryFilter added scale and shift as new arguments recently
# test with docker
# test all params once
# try and read func / class object to see kwargs
# Shift and scale are also taken as a function on BinaryFilter. Do we want to apply these even when
# do kilosort preprocessing is false? probably
# TODO: find a test case for the other annoying ones (larger recording, variable amplitude)
# TODO: test docker
# TODO: test multi-segment recording
# TODO: test do correction, skip preprocessing
# TODO: can we rename 'save_extra_kwargs' to 'save_extra_vars'. Currently untested.
# nt :  # TODO: can't kilosort figure this out from sampling rate?
# TODO: also test runtimes
# TODO: test skip preprocessing separately
# TODO: the pure default case is not tested
# TODO: shift and scale - this is also added to BinaryFilter

RUN_KILOSORT_ARGS = ["do_CAR", "invert_sign", "save_preprocessed_copy"]  # TODO: ignore some of these
# "device", "progress_bar", "save_extra_vars" are not tested. "save_extra_vars" could be.


PARAMS_TO_TEST = [
    # Not tested
    # ("torch_device", "auto")
    # Stable across KS version 4.0.01 - 4.0.12
    ("change_nothing", None),
    ("nblocks", 0),
    ("do_CAR", False),
    ("batch_size", 42743),  # Q: how much do these results change with batch size?
    ("Th_universal", 12),
    ("Th_learned", 14),
    ("invert_sign", True),
    ("nt", 93),
    ("nskip", 1),
    ("whitening_range", 16),
    ("sig_interp", 5),
    ("nt0min", 25),
    ("dmin", 15),
    ("dminx", 16),
    ("min_template_size", 15),
    ("template_sizes", 10),
    ("nearest_chans", 8),
    ("nearest_templates", 35),
    ("max_channel_distance", 5),
    ("templates_from_data", False),
    ("n_templates", 10),
    ("n_pcs", 3),
    ("Th_single_ch", 4),
    ("acg_threshold", 0.001),
    ("x_centers", 5),
    ("duplicate_spike_bins", 5),  # TODO: why is this not erroring, it is deprecated. issue on KS
    ("binning_depth", 1),
    ("artifact_threshold", 200),
    ("ccg_threshold", 1e9),
    ("cluster_downsampling", 1e9),
    ("duplicate_spike_bins", 5),  # TODO: this is depcrecated and changed to _ms in 4.0.13!
]

# Update PARAMS_TO_TEST with version-dependent kwargs
if parse(version("kilosort")) >= parse("4.0.12"):
    pass  # TODO: expose?
#    PARAMS_TO_TEST.extend(
#         [
#             ("save_preprocessed_copy", False),
#         ]
#    )
if parse(version("kilosort")) >= parse("4.0.11"):
    PARAMS_TO_TEST.extend(
        [
            ("shift", 1e9),
            ("scale", -1e9),
        ]
    )
if parse(version("kilosort")) == parse("4.0.9"):
    # bug in 4.0.9 for "nblocks=0"
    PARAMS_TO_TEST = [param for param in PARAMS_TO_TEST if param[0] != "nblocks"]

if parse(version("kilosort")) >= parse("4.0.8"):
    PARAMS_TO_TEST.extend(
        [
            ("drift_smoothing", [250, 250, 250]),
        ]
    )
if parse(version("kilosort")) <= parse("4.0.6"):
    # AFAIK this parameter was always unused in KS (that's why it was removed)
    PARAMS_TO_TEST.extend(
        [
            ("cluster_pcs", 1e9),
        ]
    )
if parse(version("kilosort")) <= parse("4.0.3"):
    PARAMS_TO_TEST = [param for param in PARAMS_TO_TEST if param[0] not in ["x_centers", "max_channel_distance"]]


class TestKilosort4Long:

    # Fixtures ######
    @pytest.fixture(scope="session")
    def recording_and_paths(self, tmp_path_factory):
        """ """
        tmp_path = tmp_path_factory.mktemp("kilosort4_tests")

        np.random.seed(0)  # TODO: check below...

        recording = self._get_ground_truth_recording()

        paths = self._save_ground_truth_recording(recording, tmp_path)

        return (recording, paths)

    @pytest.fixture(scope="session")
    def default_results(self, recording_and_paths):
        """ """
        recording, paths = recording_and_paths

        settings, ks_format_probe = self._run_kilosort_with_kilosort(recording, paths)

        defaults_ks_output_dir = paths["session_scope_tmp_path"] / "default_ks_output"

        kilosort.run_kilosort(
            settings=settings,
            probe=ks_format_probe,
            data_dtype="float32",
            results_dir=defaults_ks_output_dir,
        )

        default_results = self._get_sorting_output(defaults_ks_output_dir)

        return default_results

    # Tests ######
    def test_params_to_test(self):
        """
        Test that all parameters in PARAMS_TO_TEST are
        different than the default value used in Kilosort, otherwise
        there is no point to the test.

        TODO: need to use _default_params vs. DEFAULT_SETTINGS
        depending on decision

        TODO: write issue on this, we hope it will be on DEFAULT_SETTINGS
        TODO: duplicate_spike_ms in POSTPROCESSING but seems unused?
        """
        for parameter in PARAMS_TO_TEST:

            param_key, param_value = parameter

            if param_key == "change_nothing":
                continue

            if param_key not in RUN_KILOSORT_ARGS:
                assert DEFAULT_SETTINGS[param_key] != param_value, f"{param_key} values should be different in test."

    def test_default_settings_all_represented(self):
        """
        Test that every entry in DEFAULT_SETTINGS is tested in
        PARAMS_TO_TEST, otherwise we are missing settings added
        on the KS side.
        """
        tested_keys = [entry[0] for entry in PARAMS_TO_TEST]

        for param_key in DEFAULT_SETTINGS:

            if param_key not in ["n_chan_bin", "fs", "tmin", "tmax"]:
                assert param_key in tested_keys, f"param: {param_key} in DEFAULT SETTINGS but not tested."

    @pytest.mark.parametrize("parameter", PARAMS_TO_TEST)
    def test_kilosort4(self, recording_and_paths, default_results, tmp_path, parameter):
        """ """
        recording, paths = recording_and_paths
        param_key, param_value = parameter

        kilosort_output_dir = tmp_path / "kilosort_output_dir"
        spikeinterface_output_dir = tmp_path / "spikeinterface_output_dir"

        extra_ks_settings = {}
        if param_key == "binning_depth":
            extra_ks_settings.update({"nblocks": 5})

        if param_key in RUN_KILOSORT_ARGS:
            run_kilosort_kwargs = {param_key: param_value}
        else:
            if param_key != "change_nothing":
                extra_ks_settings.update({param_key: param_value})
            run_kilosort_kwargs = {}

        settings, ks_format_probe = self._run_kilosort_with_kilosort(recording, paths, extra_ks_settings)

        kilosort.run_kilosort(
            settings=settings,
            probe=ks_format_probe,
            data_dtype="float32",
            results_dir=kilosort_output_dir,
            **run_kilosort_kwargs,
        )

        extra_si_settings = {}
        if param_key != "change_nothing":
            extra_si_settings.update({param_key: param_value})

        if param_key == "binning_depth":
            extra_si_settings.update({"nblocks": 5})

        spikeinterface_settings = self._get_spikeinterface_settings(extra_settings=extra_si_settings)
        si.run_sorter(
            "kilosort4",
            recording,
            remove_existing_folder=True,
            folder=spikeinterface_output_dir,
            **spikeinterface_settings,
        )

        results = self._get_sorting_output(kilosort_output_dir, spikeinterface_output_dir)

        assert np.array_equal(results["ks"]["st"], results["si"]["st"]), f"{param_key} spike times different"

        assert all(
            results["ks"]["clus"].iloc[:, 0] == results["si"]["clus"].iloc[:, 0]
        ), f"{param_key} cluster assignment different"
        assert all(
            results["ks"]["clus"].iloc[:, 1] == results["si"]["clus"].iloc[:, 1]
        ), f"{param_key} cluster quality different"  # TODO: check pandas probably better way

        # This is saved on the SI side so not an extremely
        # robust addition, but it can't hurt.
        if param_key != "change_nothing":
            ops = np.load(spikeinterface_output_dir / "sorter_output" / "ops.npy", allow_pickle=True)
            ops = ops.tolist()  # strangely this makes a dict
            assert ops[param_key] == param_value

        # Finally, check out test parameters actually changes stuff!
        if parse(version("kilosort")) > parse("4.0.4"):
            self._check_test_parameters_are_actually_changing_the_output(results, default_results, param_key)

    def test_kilosort4_no_correction(self, recording_and_paths, tmp_path):
        """ """
        recording, paths = recording_and_paths

        kilosort_output_dir = tmp_path / "kilosort_output_dir"  # TODO: a lost of copying here
        spikeinterface_output_dir = tmp_path / "spikeinterface_output_dir"

        settings, ks_format_probe = self._run_kilosort_with_kilosort(recording, paths, extra_settings={"nblocks": 0})

        kilosort.run_kilosort(
            settings=settings,
            probe=ks_format_probe,
            data_dtype="float32",
            results_dir=kilosort_output_dir,
            do_CAR=True,
        )

        spikeinterface_settings = self._get_spikeinterface_settings(extra_settings={"nblocks": 6})
        si.run_sorter(
            "kilosort4",
            recording,
            remove_existing_folder=True,
            folder=spikeinterface_output_dir,
            do_correction=False,
            **spikeinterface_settings,
        )

        results = self._get_sorting_output(kilosort_output_dir, spikeinterface_output_dir)

        assert np.array_equal(results["ks"]["st"], results["si"]["st"])

        assert all(results["ks"]["clus"].iloc[:, 0] == results["si"]["clus"].iloc[:, 0])
        assert all(results["ks"]["clus"].iloc[:, 1] == results["si"]["clus"].iloc[:, 1])

    def test_kilosort4_skip_preprocessing_correction(self, tmp_path, monkeypatch):
        """ """
        recording = self._get_ground_truth_recording()

        # We need to filter and whiten the recording here to KS takes forever.
        # Do this in a way differnt to KS.
        recording = si.highpass_filter(recording, 300)
        recording = si.whiten(recording, mode="local", apply_mean=False)

        paths = self._save_ground_truth_recording(recording, tmp_path)

        kilosort_default_output_dir = tmp_path / "kilosort_default_output_dir"
        kilosort_output_dir = tmp_path / "kilosort_output_dir"
        spikeinterface_output_dir = tmp_path / "spikeinterface_output_dir"

        ks_settings, ks_format_probe = self._run_kilosort_with_kilosort(recording, paths, extra_settings={"nblocks": 0})

        kilosort.run_kilosort(
            settings=ks_settings,
            probe=ks_format_probe,
            data_dtype="float32",
            results_dir=kilosort_default_output_dir,
            do_CAR=False,
        )

        # Now the tricky bit, we need to turn off preprocessing in kilosort.
        # This is not exposed by run_kilosort() arguments (at 4.0.12 at least)
        # and so we need to monkeypatch the internal functions. The easiest
        # thing to do would be to set `get_highpass_filter()` and
        # `get_whitening_matrix()` to return `None` so these steps are skipped
        # in BinaryFilter. Unfortunately the ops saving machinery requires
        # these to be torch arrays and will error otherwise, so instead
        # we must set the filter (in frequency space) and whitening matrix
        # to unity operations so the filter and whitening do nothing. It is
        # also required to turn off motion correection to avoid some additional
        # magic KS is doing at the whitening step when motion correction is on.
        fake_filter = np.ones(60122, dtype="float32")  # TODO: hard coded
        fake_filter = torch.from_numpy(fake_filter).to("cpu")

        fake_white_matrix = np.eye(recording.get_num_channels(), dtype="float32")
        fake_white_matrix = torch.from_numpy(fake_white_matrix).to("cpu")

        def fake_fft_highpass(*args, **kwargs):
            return fake_filter

        def fake_get_whitening_matrix(*args, **kwargs):
            return fake_white_matrix

        def fake_fftshift(X, dim):
            return X

        monkeypatch.setattr("kilosort.io.fft_highpass", fake_fft_highpass)
        monkeypatch.setattr("kilosort.preprocessing.get_whitening_matrix", fake_get_whitening_matrix)
        monkeypatch.setattr("kilosort.io.fftshift", fake_fftshift)

        kilosort.run_kilosort(
            settings=ks_settings,
            probe=ks_format_probe,
            data_dtype="float32",
            results_dir=kilosort_output_dir,
            do_CAR=False,
        )

        monkeypatch.undo()

        # Now, run kilosort through spikeinterface with the same options.
        spikeinterface_settings = self._get_spikeinterface_settings(extra_settings={"nblocks": 0})
        si.run_sorter(
            "kilosort4",
            recording,
            remove_existing_folder=True,
            folder=spikeinterface_output_dir,
            do_CAR=False,
            skip_kilosort_preprocessing=True,
            **spikeinterface_settings,
        )

        default_results = self._get_sorting_output(kilosort_default_output_dir)
        results = self._get_sorting_output(kilosort_output_dir, spikeinterface_output_dir)

        # Check that out intervention actually make some difference to KS output
        # (or this test would do nothing). Then check SI and KS outputs with
        # preprocessing skipped are identical.
        assert not np.array_equal(default_results["ks"]["st"], results["ks"]["st"])
        assert np.array_equal(results["ks"]["st"], results["si"]["st"])

    # Helpers ######
    def _check_test_parameters_are_actually_changing_the_output(self, results, default_results, param_key):
        """ """
        if param_key not in ["artifact_threshold", "ccg_threshold", "cluster_downsampling"]:
            num_clus = np.unique(results["si"]["clus"].iloc[:, 0]).size
            num_clus_default = np.unique(default_results["ks"]["clus"].iloc[:, 0]).size

            if param_key == "change_nothing":
                # TODO: lol
                assert (
                    (results["si"]["st"].size == default_results["ks"]["st"].size)
                    and num_clus == num_clus_default
                    and all(results["si"]["clus"].iloc[:, 1] == default_results["ks"]["clus"].iloc[:, 1])
                ), f"{param_key} changed somehow!."
            else:
                assert (
                    (results["si"]["st"].size != default_results["ks"]["st"].size)
                    or num_clus != num_clus_default
                    or not all(results["si"]["clus"].iloc[:, 1] == default_results["ks"]["clus"].iloc[:, 1])
                ), f"{param_key} results did not change with parameter change."

    def _run_kilosort_with_kilosort(self, recording, paths, extra_settings=None):
        """ """
        # dont actually run KS here because we will overwrite the defaults!
        settings = {
            "data_dir": paths["recording_path"],
            "n_chan_bin": recording.get_num_channels(),
            "fs": recording.get_sampling_frequency(),
        }

        if extra_settings is not None:
            settings.update(extra_settings)

        ks_format_probe = load_probe(paths["probe_path"])

        return settings, ks_format_probe

    def _get_spikeinterface_settings(self, extra_settings=None):
        """ """
        # dont actually run here.
        settings = copy.deepcopy(DEFAULT_SETTINGS)

        if extra_settings is not None:
            settings.update(extra_settings)

        for name in ["n_chan_bin", "fs", "tmin", "tmax"]:  # TODO: check tmin and tmax
            settings.pop(name)

        return settings

    def _get_sorting_output(self, kilosort_output_dir=None, spikeinterface_output_dir=None) -> dict[str, Any]:
        """ """
        results = {
            "si": {},
            "ks": {},
        }
        if kilosort_output_dir:
            results["ks"]["st"] = np.load(kilosort_output_dir / "spike_times.npy")
            results["ks"]["clus"] = pd.read_table(kilosort_output_dir / "cluster_group.tsv")

        if spikeinterface_output_dir:
            results["si"]["st"] = np.load(spikeinterface_output_dir / "sorter_output" / "spike_times.npy")
            results["si"]["clus"] = pd.read_table(spikeinterface_output_dir / "sorter_output" / "cluster_group.tsv")

        return results

    def _get_ground_truth_recording(self):
        """ """
        # Chosen so all parameter changes to indeed change the output
        num_channels = 32
        recording, _ = si.generate_ground_truth_recording(
            durations=[5],
            seed=0,
            num_channels=num_channels,
            num_units=5,
            generate_sorting_kwargs=dict(firing_rates=100, refractory_period_ms=4.0),
        )
        return recording

    def _save_ground_truth_recording(self, recording, tmp_path):
        """ """
        paths = {
            "session_scope_tmp_path": tmp_path,
            "recording_path": tmp_path / "my_test_recording",
            "probe_path": tmp_path / "my_test_probe.prb",
        }

        recording.save(folder=paths["recording_path"], overwrite=True)

        probegroup = recording.get_probegroup()
        write_prb(paths["probe_path"].as_posix(), probegroup)

        return paths
