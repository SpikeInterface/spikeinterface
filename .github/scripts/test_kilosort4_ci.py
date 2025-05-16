"""
This file tests the SpikeInterface wrapper of the Kilosort4. The general logic
of the tests are:
- Change every exposed parameter one at a time (PARAMS_TO_TEST). Check that
  the result of the SpikeInterface wrapper and Kilosort run natively are
  the same. The SpikeInterface wrapper is non-trivial and decomposes the
  kilosort pipeline to allow additions such as skipping preprocessing. Therefore,
  the idea is that is it safer to rely on the output directly rather than
  try monkeypatching. One thing can could be better tested is parameter
  changes when skipping KS4 preprocessing is true, because this takes a slightly
  different path through the kilosort4.py wrapper logic.
  This also checks that changing the parameter changes the test output from default
  on our test case (otherwise, the test could not detect a failure).

- Test that kilosort functions called from `kilosort4.py` wrapper have the expected
  input signatures

- Do some tests to check all KS4 parameters are tested against.
"""

import pytest
import copy
from packaging.version import parse
from inspect import signature

import numpy as np
import torch

import spikeinterface.full as si
from spikeinterface.core.testing import check_sortings_equal
from spikeinterface.sorters.external.kilosort4 import Kilosort4Sorter
from probeinterface.io import write_prb

import kilosort
from kilosort.parameters import DEFAULT_SETTINGS
from kilosort.run_kilosort import (
    set_files,
    initialize_ops,
    compute_preprocessing,
    compute_drift_correction,
    detect_spikes,
    cluster_spikes,
    save_sorting,
    get_run_parameters,
)
from kilosort.io import load_probe, RecordingExtractorAsArray, BinaryFiltered


RUN_KILOSORT_ARGS = ["do_CAR", "invert_sign", "save_preprocessed_copy"]
# "device", "progress_bar", "save_extra_vars" are not tested. "save_extra_vars" could be.

# Setup Params to test ####
PARAMS_TO_TEST_DICT = {
    "nblocks": 0,
    "do_CAR": False,
    "batch_size": 42743,
    "Th_universal": 12,
    "Th_learned": 14,
    "invert_sign": True,
    "nt": 93,
    "nskip": 1,
    "whitening_range": 16,
    "highpass_cutoff": 200,
    "sig_interp": 5,
    "nt0min": 25,
    "dmin": 15,
    "dminx": 16,
    "min_template_size": 15,
    "template_sizes": 10,
    "nearest_chans": 8,
    "nearest_templates": 35,
    "max_channel_distance": 5,
    "n_templates": 10,
    "n_pcs": 3,
    "Th_single_ch": 4,
    "x_centers": 5,
    "binning_depth": 1,
    "drift_smoothing": [250, 250, 250],
    "artifact_threshold": 200,
    "ccg_threshold": 1e12,
    "acg_threshold": 1e12,
    "cluster_downsampling": 2,
    "duplicate_spike_ms": 0.3,
}

PARAMETERS_NOT_AFFECTING_RESULTS = [
    "artifact_threshold",
    "ccg_threshold",
    "acg_threshold",
    "cluster_downsampling",
    "cluster_pcs",
    "duplicate_spike_ms",  # this is because ground-truth spikes don't have violations
]


# Add/Remove version specific parameters
if parse(kilosort.__version__) >= parse("4.0.22"):
    PARAMS_TO_TEST_DICT.update(
        {"position_limit": 50}
    )
    # Position limit only affects computing spike locations after sorting
    PARAMETERS_NOT_AFFECTING_RESULTS.append("position_limit")

if parse(kilosort.__version__) >= parse("4.0.24"):
    PARAMS_TO_TEST_DICT.update(
        {"max_peels": 200},
    )
    # max_peels is not affecting the results in this short dataset
    PARAMETERS_NOT_AFFECTING_RESULTS.append("max_peels")

if parse(kilosort.__version__) >= parse("4.0.33"):
    PARAMS_TO_TEST_DICT.update({"cluster_neighbors": 11})
    PARAMETERS_NOT_AFFECTING_RESULTS.append("cluster_neighbors")


PARAMS_TO_TEST = list(PARAMS_TO_TEST_DICT.keys())


class TestKilosort4Long:
    # Fixtures ######
    @pytest.fixture(scope="session")
    def recording_and_paths(self, tmp_path_factory):
        """
        Create a ground-truth recording, and save it to binary
        so KS4 can run on it. Fixture is set up once and shared between
        all tests.
        """
        tmp_path = tmp_path_factory.mktemp("kilosort4_tests")

        recording = self._get_ground_truth_recording()

        paths = self._save_ground_truth_recording(recording, tmp_path)

        return (recording, paths)

    @pytest.fixture(scope="session")
    def default_kilosort_sorting(self, recording_and_paths):
        """
        Because we check each parameter at a time and check the
        KS4 and SpikeInterface versions match, if changing the parameter
        had no effect as compared to default then the test would not test
        anything. Therefore, the default results are run once and stored,
        to check changing params indeed changes the results during testing.
        This is possibly for nearly all parameters.
        """
        recording, paths = recording_and_paths

        settings, _, ks_format_probe = self._get_kilosort_native_settings(recording, paths, None, None)

        defaults_ks_output_dir = paths["session_scope_tmp_path"] / "default_ks_output"

        kilosort.run_kilosort(
            settings=settings,
            probe=ks_format_probe,
            data_dtype="float32",
            results_dir=defaults_ks_output_dir,
        )

        return si.read_kilosort(defaults_ks_output_dir)

    def _get_ground_truth_recording(self):
        """
        A ground truth recording chosen to be as small as possible (for speed).
        But contain enough information so that changing most parameters
        changes the results.
        """
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
        """
        Save the recording and its probe to file, so it can be
        loaded by KS4.
        """
        paths = {
            "session_scope_tmp_path": tmp_path,
            "recording_path": tmp_path / "my_test_recording" / "traces_cached_seg0.raw",
            "probe_path": tmp_path / "my_test_probe.prb",
        }

        recording.save(folder=paths["recording_path"].parent, overwrite=True)

        probegroup = recording.get_probegroup()
        write_prb(paths["probe_path"].as_posix(), probegroup)

        return paths

    # Tests ######
    def test_params_to_test(self):
        """
        Test that all values in PARAMS_TO_TEST_DICT are
        different to the default values used in Kilosort,
        otherwise there is no point to the test.
        """
        for param_key, param_value in PARAMS_TO_TEST_DICT.items():
            if param_key not in RUN_KILOSORT_ARGS:
                assert DEFAULT_SETTINGS[param_key] != param_value, (
                    f"{param_key} values should be different in test: "
                    f"{param_value} vs. {DEFAULT_SETTINGS[param_key]}"
                )

    def test_default_settings_all_represented(self):
        """
        Test that every entry in DEFAULT_SETTINGS is tested in
        PARAMS_TO_TEST, otherwise we are missing settings added
        on the KS side.
        """
        tested_keys = PARAMS_TO_TEST
        additional_non_tested_keys = ["shift", "scale", "save_preprocessed_copy"]
        tested_keys += additional_non_tested_keys

        for param_key in DEFAULT_SETTINGS:
            if param_key not in ["n_chan_bin", "fs", "tmin", "tmax", "templates_from_data"]:
                assert param_key in tested_keys, f"param: {param_key} in DEFAULT SETTINGS but not tested."

    def test_spikeinterface_defaults_against_kilsort(self):
        """
        Here check that all _
        Don't check that every default in KS is exposed in params,
        because they change across versions. Instead, this check
        is performed here against PARAMS_TO_TEST.
        """
        params = copy.deepcopy(Kilosort4Sorter._default_params)

        for key in params.keys():
            # "artifact threshold" is set to `np.inf` if `None` in
            # the body of the `Kilosort4Sorter` class.
            if key in DEFAULT_SETTINGS and key not in ["artifact_threshold"]:
                assert params[key] == DEFAULT_SETTINGS[key], f"{key} is not the same across versions."

    # Testing Arguments ###
    def test_set_files_arguments(self):
        expected_arguments = ["settings", "filename", "probe", "probe_name", "data_dir", "results_dir", "bad_channels"]
        if parse(kilosort.__version__) >= parse("4.0.34"):
            expected_arguments += ["shank_idx"]
        self._check_arguments(
            set_files, expected_arguments
        )

    def test_initialize_ops_arguments(self):
        expected_arguments = [
            "settings",
            "probe",
            "data_dtype",
            "do_CAR",
            "invert_sign",
            "device",
            "save_preprocessed_copy",
        ]

        self._check_arguments(
            initialize_ops,
            expected_arguments,
        )

    def test_compute_preprocessing_arguments(self):
        self._check_arguments(compute_preprocessing, ["ops", "device", "tic0", "file_object"])

    def test_compute_drift_location_arguments(self):
        expected_arguments = ["ops", "device", "tic0", "progress_bar", "file_object", "clear_cache"]
        if parse(kilosort.__version__) >= parse("4.0.28"):
            expected_arguments += ["verbose"]
        self._check_arguments(compute_drift_correction, expected_arguments)

    def test_detect_spikes_arguments(self):
        expected_arguments = ["ops", "device", "bfile", "tic0", "progress_bar", "clear_cache"]
        if parse(kilosort.__version__) >= parse("4.0.28"):
            expected_arguments += ["verbose"]
        self._check_arguments(detect_spikes, expected_arguments)

    def test_cluster_spikes_arguments(self):
        expected_arguments = ["st", "tF", "ops", "device", "bfile", "tic0", "progress_bar", "clear_cache"]
        if parse(kilosort.__version__) >= parse("4.0.28"):
            expected_arguments += ["verbose"]
        self._check_arguments(cluster_spikes, expected_arguments)

    def test_save_sorting_arguments(self):
        expected_arguments = ["ops", "results_dir", "st", "clu", "tF", "Wall", "imin", "tic0", "save_extra_vars"]

        expected_arguments.append("save_preprocessed_copy")

        self._check_arguments(save_sorting, expected_arguments)

    def test_get_run_parameters(self):
        self._check_arguments(get_run_parameters, ["ops"])

    def test_load_probe_parameters(self):
        self._check_arguments(load_probe, ["probe_path"])

    def test_recording_extractor_as_array_arguments(self):
        self._check_arguments(RecordingExtractorAsArray, ["recording_extractor"])

    def test_binary_filtered_arguments(self):
        expected_arguments = [
            "filename",
            "n_chan_bin",
            "fs",
            "NT",
            "nt",
            "nt0min",
            "chan_map",
            "hp_filter",
            "whiten_mat",
            "dshift",
            "device",
            "do_CAR",
            "artifact_threshold",
            "invert_sign",
            "dtype",
            "tmin",
            "tmax",
            "shift",
            "scale",
            "file_object",
        ]

        self._check_arguments(BinaryFiltered, expected_arguments)

    def _check_arguments(self, object_, expected_arguments):
        """
        Check that the argument signature of  `object_` is as expected
        (i.e. has not changed across kilosort versions).
        """
        sig = signature(object_)
        obj_arguments = list(sig.parameters.keys())
        assert expected_arguments == obj_arguments

    # Full Test ####
    @pytest.mark.parametrize("parameter", PARAMS_TO_TEST)
    def test_kilosort4_main(self, recording_and_paths, default_kilosort_sorting, tmp_path, parameter):
        """
        Given a recording, paths to raw data, and a parameter to change,
        run KS4 natively and within the SpikeInterface wrapper with the
        new parameter value (all other values default) and
        check the outputs are the same.
        """
        recording, paths = recording_and_paths
        param_key = parameter
        param_value = PARAMS_TO_TEST_DICT[param_key]

        # Setup parameters for KS4 and run it natively
        kilosort_output_dir = tmp_path / "kilosort_output_dir"
        spikeinterface_output_dir = tmp_path / "spikeinterface_output_dir"

        settings, run_kilosort_kwargs, ks_format_probe = self._get_kilosort_native_settings(
            recording, paths, param_key, param_value
        )

        kilosort.run_kilosort(
            settings=settings,
            probe=ks_format_probe,
            data_dtype="float32",
            results_dir=kilosort_output_dir,
            **run_kilosort_kwargs,
        )
        sorting_ks4 = si.read_kilosort(kilosort_output_dir)

        # Setup Parameters for SI and KS4 through SI
        spikeinterface_settings = self._get_spikeinterface_settings(param_key, param_value)

        sorting_si = si.run_sorter(
            "kilosort4",
            recording,
            remove_existing_folder=True,
            folder=spikeinterface_output_dir,
            **spikeinterface_settings,
        )

        # Get the results and check they match
        check_sortings_equal(sorting_ks4, sorting_si)

        # Check the ops file in KS4 output is as expected. This is saved on the
        # SI side so not an extremely robust addition, but it can't hurt.
        ops = np.load(spikeinterface_output_dir / "sorter_output" / "ops.npy", allow_pickle=True)
        ops = ops.tolist()  # strangely this makes a dict
        assert ops[param_key] == param_value

        # Finally, check out test parameters actually change the output of
        # KS4, ensuring our tests are actually doing something (exxcept for some params).
        if param_key not in PARAMETERS_NOT_AFFECTING_RESULTS:
            with pytest.raises(AssertionError):
                check_sortings_equal(default_kilosort_sorting, sorting_si)

    def test_clear_cache(self,recording_and_paths, tmp_path):
        """
        Test clear_cache parameter in kilosort4.run_kilosort
        """
        recording, paths = recording_and_paths

        spikeinterface_output_dir = tmp_path / "spikeinterface_output_clear"
        sorting_si_clear = si.run_sorter(
            "kilosort4",
            recording,
            remove_existing_folder=True,
            folder=spikeinterface_output_dir,
            clear_cache=True
        )
        spikeinterface_output_dir = tmp_path / "spikeinterface_output_no_clear"
        sorting_si_no_clear = si.run_sorter(
            "kilosort4",
            recording,
            remove_existing_folder=True,
            folder=spikeinterface_output_dir,
            clear_cache=False
        )
        check_sortings_equal(sorting_si_clear, sorting_si_no_clear)

    def test_kilosort4_no_correction(self, recording_and_paths, tmp_path):
        """
        Test the SpikeInterface wrappers `do_correction` argument. We set
        `nblocks=0` for KS4 native, turning off motion correction. Then
        we run KS$ through SpikeInterface with `do_correction=False` but
        `nblocks=1` (KS4 default) - checking that `do_correction` overrides
        this and the result matches KS4 when run without motion correction.
        """
        recording, paths = recording_and_paths

        kilosort_output_dir = tmp_path / "kilosort_output_dir"
        spikeinterface_output_dir = tmp_path / "spikeinterface_output_dir"

        settings, _, ks_format_probe = self._get_kilosort_native_settings(recording, paths, "nblocks", 0)

        kilosort.run_kilosort(
            settings=settings,
            probe=ks_format_probe,
            data_dtype="float32",
            results_dir=kilosort_output_dir,
            do_CAR=True,
        )
        sorting_ks = si.read_kilosort(kilosort_output_dir)

        spikeinterface_settings = self._get_spikeinterface_settings("nblocks", 1)
        sorting_si = si.run_sorter(
            "kilosort4",
            recording,
            remove_existing_folder=True,
            folder=spikeinterface_output_dir,
            do_correction=False,
            **spikeinterface_settings,
        )
        check_sortings_equal(sorting_ks, sorting_si)

    def test_use_binary_file(self, tmp_path):
        """
        Test that the SpikeInterface wrapper can run KS4 using a binary file as input or directly
        from the recording.
        """
        recording = self._get_ground_truth_recording()
        recording_bin = recording.save()

        # run with SI wrapper
        sorting_ks4 = si.run_sorter(
            "kilosort4",
            recording,
            folder=tmp_path / "ks4_output_si_wrapper_default",
            use_binary_file=None,
            remove_existing_folder=True,
        )
        sorting_ks4_bin = si.run_sorter(
            "kilosort4",
            recording_bin,
            folder=tmp_path / "ks4_output_bin_default",
            use_binary_file=None,
            remove_existing_folder=True,
        )
        sorting_ks4_force_binary = si.run_sorter(
            "kilosort4",
            recording,
            folder=tmp_path / "ks4_output_force_bin",
            use_binary_file=True,
            remove_existing_folder=True,
        )
        assert not (tmp_path / "ks4_output_force_bin" / "sorter_output" / "recording.dat").exists()
        sorting_ks4_force_non_binary = si.run_sorter(
            "kilosort4",
            recording_bin,
            folder=tmp_path / "ks4_output_force_wrapper",
            use_binary_file=False,
            remove_existing_folder=True,
        )
        # test deleting recording.dat
        sorting_ks4_force_binary_keep = si.run_sorter(
            "kilosort4",
            recording,
            folder=tmp_path / "ks4_output_force_bin_keep",
            use_binary_file=True,
            delete_recording_dat=False,
            remove_existing_folder=True,
        )
        assert (tmp_path / "ks4_output_force_bin_keep" / "sorter_output" / "recording.dat").exists()

        check_sortings_equal(sorting_ks4, sorting_ks4_bin)
        check_sortings_equal(sorting_ks4, sorting_ks4_force_binary)
        check_sortings_equal(sorting_ks4, sorting_ks4_force_non_binary)

    @pytest.mark.parametrize(
        "param_to_test",
        [
            ("do_CAR", False),
            ("batch_size", 42743),
            ("Th_learned", 14),
            ("dmin", 15),
            ("max_channel_distance", 5),
            ("n_pcs", 3),
        ],
    )
    def test_kilosort4_skip_preprocessing_correction(self, tmp_path, monkeypatch, param_to_test):
        """
        Test that skipping KS4 preprocessing works as expected. Run
        KS4 natively, monkeypatching the relevant preprocessing functions
        such that preprocessing is not performed. Then run in SpikeInterface
        with `skip_kilosort_preprocessing=True` and check the outputs match.

        Run with a few randomly chosen parameters to check these are propagated
        under this condition.

        TODO
        ----
        It would be nice to check a few additional parameters here. Screw it!
        """
        param_key, param_value = param_to_test

        recording = self._get_ground_truth_recording()

        # We need to filter and whiten the recording here to KS takes forever.
        # Do this in a way different to KS.
        recording = si.highpass_filter(recording, 300)
        recording = si.whiten(recording, mode="local", apply_mean=False)

        paths = self._save_ground_truth_recording(recording, tmp_path)

        kilosort_output_dir = tmp_path / "kilosort_output_dir"
        spikeinterface_output_dir = tmp_path / "spikeinterface_output_dir"

        if parse(kilosort.__version__) >= parse("4.0.33"):
            def monkeypatch_filter_function(self, X, ops=None, ibatch=None, skip_preproc=False):
                """
                This is a direct copy of the kilosort io.BinaryFiltered.filter
                function, with hp_filter and whitening matrix code sections, and
                comments removed. This is the easiest way to monkeypatch (tried a few approaches)
                """
                if self.chan_map is not None:
                    X = X[self.chan_map]

                if self.invert_sign:
                    X = X * -1

                X = X - X.mean(1).unsqueeze(1)
                if self.do_CAR:
                    X = X - torch.median(X, 0)[0]

                if self.hp_filter is not None:
                    pass

                if self.artifact_threshold < np.inf:
                    if torch.any(torch.abs(X) >= self.artifact_threshold):
                        return torch.zeros_like(X)

                if self.whiten_mat is not None:
                    pass
                return X
        else:
            def monkeypatch_filter_function(self, X, ops=None, ibatch=None):
                """
                This is a direct copy of the kilosort io.BinaryFiltered.filter
                function, with hp_filter and whitening matrix code sections, and
                comments removed. This is the easiest way to monkeypatch (tried a few approaches)
                """
                if self.chan_map is not None:
                    X = X[self.chan_map]

                if self.invert_sign:
                    X = X * -1

                X = X - X.mean(1).unsqueeze(1)
                if self.do_CAR:
                    X = X - torch.median(X, 0)[0]

                if self.hp_filter is not None:
                    pass

                if self.artifact_threshold < np.inf:
                    if torch.any(torch.abs(X) >= self.artifact_threshold):
                        return torch.zeros_like(X)

                if self.whiten_mat is not None:
                    pass
                return X
        monkeypatch.setattr("kilosort.io.BinaryFiltered.filter", monkeypatch_filter_function)

        ks_settings, _, ks_format_probe = self._get_kilosort_native_settings(recording, paths, param_key, param_value)
        ks_settings["nblocks"] = 0

        # Be explicit here and don't rely on defaults.
        do_CAR = param_value if param_key == "do_CAR" else False

        kilosort.run_kilosort(
            settings=ks_settings,
            probe=ks_format_probe,
            data_dtype="float32",
            results_dir=kilosort_output_dir,
            do_CAR=do_CAR,
        )

        monkeypatch.undo()
        si.read_kilosort(kilosort_output_dir)

        # Now, run kilosort through spikeinterface with the same options.
        spikeinterface_settings = self._get_spikeinterface_settings(param_key, param_value)
        spikeinterface_settings["nblocks"] = 0

        do_CAR = False if param_key != "do_CAR" else spikeinterface_settings.pop("do_CAR")

        si.run_sorter(
            "kilosort4",
            recording,
            remove_existing_folder=True,
            folder=spikeinterface_output_dir,
            do_CAR=do_CAR,
            skip_kilosort_preprocessing=True,
            **spikeinterface_settings,
        )

        # There is a very slight difference caused by the batching between load vs.
        # memory file. Because in this test recordings are preprocessed, there are
        # some filter edge effects that depend on the chunking in `get_traces()`.
        # These are all extremely close (usually just 1 spike, 1 idx different).
        results = {}
        results["ks"] = {}
        results["ks"]["st"] = np.load(kilosort_output_dir / "spike_times.npy")
        results["ks"]["clus"] = np.load(kilosort_output_dir / "spike_clusters.npy")
        results["si"] = {}
        results["si"]["st"] = np.load(spikeinterface_output_dir / "sorter_output" / "spike_times.npy")
        results["si"]["clus"] = np.load(spikeinterface_output_dir / "sorter_output" / "spike_clusters.npy")
        assert np.allclose(results["ks"]["st"], results["si"]["st"], rtol=0, atol=1)
        assert np.array_equal(results["ks"]["clus"], results["si"]["clus"])

    ##### Helpers ######
    def _get_kilosort_native_settings(self, recording, paths, param_key, param_value):
        """
        Function to generate the settings and function inputs to run kilosort.
        Note when `binning_depth` is used we need to set `nblocks` high to
        get the results to change from default.

        Some settings in KS4 are passed by `settings` dict while others
        are through the function, these are split here.
        """
        settings = {
            "filename": paths["recording_path"],
            "n_chan_bin": recording.get_num_channels(),
            "fs": recording.get_sampling_frequency(),
        }
        run_kilosort_kwargs = {}

        if param_key is not None:
            if param_key == "binning_depth":
                settings.update({"nblocks": 5})

            if param_key in RUN_KILOSORT_ARGS:
                run_kilosort_kwargs = {param_key: param_value}
            else:
                settings.update({param_key: param_value})
                run_kilosort_kwargs = {}

        ks_format_probe = load_probe(paths["probe_path"])

        return settings, run_kilosort_kwargs, ks_format_probe

    def _get_spikeinterface_settings(self, param_key, param_value):
        """
        Generate settings kwargs for running KS4 in SpikeInterface.
        See `_get_kilosort_native_settings()` for some details.
        """
        settings = {}  # copy.deepcopy(DEFAULT_SETTINGS)

        if param_key == "binning_depth":
            settings.update({"nblocks": 5})

        settings.update({param_key: param_value})

        # for name in ["n_chan_bin", "fs", "tmin", "tmax"]:
        #    settings.pop(name)

        return settings
