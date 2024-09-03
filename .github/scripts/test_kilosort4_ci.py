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
  on our test case (otherwise, the test could not detect a failure). This is possible
  for nearly all parameters, see `_check_test_parameters_are_changing_the_output()`.

- Test that kilosort functions called from `kilosort4.py` wrapper have the expected
  input signatures

- Do some tests to check all KS4 parameters are tested against.
"""

import copy
from typing import Any
import numpy as np
import torch
import kilosort
from kilosort.io import load_probe
import pandas as pd
import pytest
from packaging.version import parse
from importlib.metadata import version
from inspect import signature

import spikeinterface.full as si
from spikeinterface.sorters.external.kilosort4 import Kilosort4Sorter
from probeinterface.io import write_prb

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
from kilosort.parameters import DEFAULT_SETTINGS
from kilosort import preprocessing as ks_preprocessing

RUN_KILOSORT_ARGS = ["do_CAR", "invert_sign", "save_preprocessed_copy"]
# "device", "progress_bar", "save_extra_vars" are not tested. "save_extra_vars" could be.

# Setup Params to test ####
PARAMS_TO_TEST = [
    # Not tested
    # ("torch_device", "auto")
    # Stable across KS version 4.0.16 - 4.0.X (?)
    ("change_nothing", None),
    ("nblocks", 0),
    ("do_CAR", False),
    ("batch_size", 42743),
    ("Th_universal", 12),
    ("Th_learned", 14),
    ("invert_sign", True),
    ("nt", 93),
    ("nskip", 1),
    ("whitening_range", 16),
    ("highpass_cutoff", 200),
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
    ("x_centers", 5),
    ("binning_depth", 1),
    # Note: These don't change the results from
    # default when applied to the test case.
    ("artifact_threshold", 200),
    ("ccg_threshold", 1e12),
    ("acg_threshold", 1e12),
    ("cluster_downsampling", 2),
    ("duplicate_spike_ms", 0.3),
    ("drift_smoothing", [250, 250, 250]),
    ("save_preprocessed_copy", False),
    ("shift", 0),
    ("scale", 1),
]


# if parse(version("kilosort")) >= parse("4.0.X"):
#     PARAMS_TO_TEST.extend(
#         [
#             ("new_param", new_values),
#         ]
#     )


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
    def default_results(self, recording_and_paths):
        """
        Because we check each parameter at a time and check the
        KS4 and SpikeInterface versions match, if changing the parameter
        had no effect as compared to default then the test would not test
        anything. Therefore, the default results are run once and stored,
        to check changing params indeed changes the results during testing.
        This is possibly for nearly all parameters.
        """
        recording, paths = recording_and_paths

        settings, _, ks_format_probe = self._get_kilosort_native_settings(recording, paths, "change_nothing", None)

        defaults_ks_output_dir = paths["session_scope_tmp_path"] / "default_ks_output"

        kilosort.run_kilosort(
            settings=settings,
            probe=ks_format_probe,
            data_dtype="float32",
            results_dir=defaults_ks_output_dir,
        )

        default_results = self._get_sorting_output(defaults_ks_output_dir)

        return default_results

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
            "recording_path": tmp_path / "my_test_recording",
            "probe_path": tmp_path / "my_test_probe.prb",
        }

        recording.save(folder=paths["recording_path"], overwrite=True)

        probegroup = recording.get_probegroup()
        write_prb(paths["probe_path"].as_posix(), probegroup)

        return paths

    # Tests ######
    def test_params_to_test(self):
        """
        Test that all values in PARAMS_TO_TEST are
        different to the default values used in Kilosort,
        otherwise there is no point to the test.
        """
        for parameter in PARAMS_TO_TEST:
            param_key, param_value = parameter

            if param_key == "change_nothing":
                continue

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
        tested_keys = [entry[0] for entry in PARAMS_TO_TEST]

        for param_key in DEFAULT_SETTINGS:
            if param_key not in ["n_chan_bin", "fs", "tmin", "tmax"]:
                if parse(version("kilosort")) == parse("4.0.9") and param_key == "nblocks":
                    continue
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
        self._check_arguments(set_files, ["settings", "filename", "probe", "probe_name", "data_dir", "results_dir", "bad_channels"])

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
        self._check_arguments(compute_drift_correction, ["ops", "device", "tic0", "progress_bar", "file_object", "clear_cache"])

    def test_detect_spikes_arguments(self):
        self._check_arguments(detect_spikes, ["ops", "device", "bfile", "tic0", "progress_bar", "clear_cache"])

    def test_cluster_spikes_arguments(self):
        self._check_arguments(cluster_spikes, ["st", "tF", "ops", "device", "bfile", "tic0", "progress_bar", "clear_cache"])

    def test_save_sorting_arguments(self):
        expected_arguments = ["ops", "results_dir", "st", "clu", "tF", "Wall", "imin", "tic0", "save_extra_vars"]

        if parse(version("kilosort")) > parse("4.0.11"):
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
    def test_kilosort4_main(self, recording_and_paths, default_results, tmp_path, parameter):
        """
        Given a recording, paths to raw data, and a parameter to change,
        run KS4 natively and within the SpikeInterface wrapper with the
        new parameter value (all other values default) and
        check the outputs are the same.
        """
        recording, paths = recording_and_paths
        param_key, param_value = parameter

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

        # Setup Parameters for SI and KS4 through SI
        spikeinterface_settings = self._get_spikeinterface_settings(param_key, param_value)

        si.run_sorter(
            "kilosort4",
            recording,
            remove_existing_folder=True,
            folder=spikeinterface_output_dir,
            **spikeinterface_settings,
        )

        # Get the results and check they match
        results = self._get_sorting_output(kilosort_output_dir, spikeinterface_output_dir)

        assert np.array_equal(results["ks"]["st"], results["si"]["st"]), f"{param_key} spike times different"
        assert np.array_equal(results["ks"]["clus"], results["si"]["clus"]), f"{param_key} cluster assignment different"

        # Check the ops file in KS4 output is as expected. This is saved on the
        # SI side so not an extremely robust addition, but it can't hurt.
        if param_key != "change_nothing":
            ops = np.load(spikeinterface_output_dir / "sorter_output" / "ops.npy", allow_pickle=True)
            ops = ops.tolist()  # strangely this makes a dict
            assert ops[param_key] == param_value

        # Finally, check out test parameters actually change the output of
        # KS4, ensuring our tests are actually doing something. This is not
        # done prior to 4.0.4 because a number of parameters seem to stop
        # having an effect. This is probably due to small changes in their
        # behaviour, and the test file chosen here.
        if parse(version("kilosort")) > parse("4.0.4"):
            self._check_test_parameters_are_changing_the_output(results, default_results, param_key)

    @pytest.mark.skipif(parse(version("kilosort")) == parse("4.0.9"), reason="nblock=0 fails on KS4=4.0.9")
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

        spikeinterface_settings = self._get_spikeinterface_settings("nblocks", 1)
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
        assert np.array_equal(results["ks"]["clus"], results["si"]["clus"])

    @pytest.mark.skipif(parse(version("kilosort")) == parse("4.0.9"), reason="nblock=0 fails on KS4=4.0.9")
    @pytest.mark.parametrize(
        "param_to_test",
        [
            ("change_nothing", None),
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
        results = self._get_sorting_output(kilosort_output_dir, spikeinterface_output_dir)
        assert np.allclose(results["ks"]["st"], results["si"]["st"], rtol=0, atol=1)

    # Helpers ######
    def _check_test_parameters_are_changing_the_output(self, results, default_results, param_key):
        """
        If nothing is changed, default vs. results outputs are identical.
        Otherwise, check they are not the same. Can't figure out how to get
        the skipped three parameters below to change the results on this
        small test file.
        """
        if param_key in ["acg_threshold", "ccg_threshold", "artifact_threshold", "cluster_downsampling", "cluster_pcs"]:
            return

        if param_key == "change_nothing":
            assert all(default_results["ks"]["st"] == results["ks"]["st"]) and all(
                default_results["ks"]["clus"] == results["ks"]["clus"]
            ), f"{param_key} changed somehow!."
        else:
            assert not (default_results["ks"]["st"].size == results["ks"]["st"].size) or not all(
                default_results["ks"]["clus"] == results["ks"]["clus"]
            ), f"{param_key} results did not change with parameter change."

    def _get_kilosort_native_settings(self, recording, paths, param_key, param_value):
        """
        Function to generate the settings and function inputs to run kilosort.
        Note when `binning_depth` is used we need to set `nblocks` high to
        get the results to change from default.

        Some settings in KS4 are passed by `settings` dict while others
        are through the function, these are split here.
        """
        settings = {
            "data_dir": paths["recording_path"],
            "n_chan_bin": recording.get_num_channels(),
            "fs": recording.get_sampling_frequency(),
        }

        if param_key == "binning_depth":
            settings.update({"nblocks": 5})

        if param_key in RUN_KILOSORT_ARGS:
            run_kilosort_kwargs = {param_key: param_value}
        else:
            if param_key != "change_nothing":
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

        if param_key != "change_nothing":
            settings.update({param_key: param_value})

        if param_key == "binning_depth":
            settings.update({"nblocks": 5})

        # for name in ["n_chan_bin", "fs", "tmin", "tmax"]:
        #    settings.pop(name)

        return settings

    def _get_sorting_output(self, kilosort_output_dir=None, spikeinterface_output_dir=None) -> dict[str, Any]:
        """
        Load the results of sorting into a dict for easy comparison.
        """
        results = {
            "si": {},
            "ks": {},
        }
        if kilosort_output_dir:
            results["ks"]["st"] = np.load(kilosort_output_dir / "spike_times.npy")
            results["ks"]["clus"] = np.load(kilosort_output_dir / "spike_clusters.npy")

        if spikeinterface_output_dir:
            results["si"]["st"] = np.load(spikeinterface_output_dir / "sorter_output" / "spike_times.npy")
            results["si"]["clus"] = np.load(spikeinterface_output_dir / "sorter_output" / "spike_clusters.npy")

        return results
