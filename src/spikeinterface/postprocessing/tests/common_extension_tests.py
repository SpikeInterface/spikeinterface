from __future__ import annotations

import pytest
import shutil
import numpy as np

from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.core import create_sorting_analyzer
from spikeinterface.core import estimate_sparsity


def get_dataset():
    recording, sorting = generate_ground_truth_recording(
        durations=[15.0, 5.0],
        sampling_frequency=24000.0,
        num_channels=6,
        num_units=3,
        generate_sorting_kwargs=dict(firing_rates=3.0, refractory_period_ms=4.0),
        generate_unit_locations_kwargs=dict(
            margin_um=5.0,
            minimum_z=5.0,
            maximum_z=20.0,
        ),
        generate_templates_kwargs=dict(
            unit_params=dict(
                alpha=(100.0, 500.0),
            )
        ),
        noise_kwargs=dict(noise_levels=5.0, strategy="tile_pregenerated"),
        seed=2205,
    )
    return recording, sorting


class AnalyzerExtensionCommonTestSuite:
    """
    Common tests with class approach to compute extension on several cases,
    format ("memory", "binary_folder", "zarr") and sparsity (True, False).
    Extensions refer to the extension classes that handle the postprocessing,
    for example extracting principal components or amplitude scalings.

    This base class provides a fixture which sets a recording
    and sorting object onto itself, which are set up once each time
    the base class is subclassed in a test environment. The recording
    and sorting object are used in the creation of the `sorting_analyzer`
    object used to run postprocessing routines.

    When subclassed, a test function that parametrises arguments
    that are passed to the `sorting_analyzer.compute()` can be setup.
    This must call `run_extension_tests()`  which sets up a `sorting_analyzer`
    with the relevant format and sparsity. This also automatically precomputes
    extension dependencies with default params, Then, `check_one()` is called
    which runs the compute function with the passed params and tests that:

    1) the returned extractor object has data on it
    2) check `sorting_analyzer.get_extension()` does not return None
    3) the correct units are sliced with the `select_units()` function.
    """

    @pytest.fixture(autouse=True, scope="class")
    def setUpClass(self, create_cache_folder):
        """
        This method sets up the class once at the start of testing. It is
        in scope for the lifetime of te class and is reused across all
        tests that inherit from this base class to save processing time and
        force a small radius.

        When setting attributes on `self` in `scope="class"` a new
        class instance is used for each. In this case, we have to set
        from the base object `__class__` to ensure the attributes
        are available to all subclass instances.
        """
        self.__class__.recording, self.__class__.sorting = get_dataset()

        self.__class__.sparsity = estimate_sparsity(
            self.__class__.sorting, self.__class__.recording, method="radius", radius_um=20
        )
        self.__class__.cache_folder = create_cache_folder

    def get_sorting_analyzer(self, recording, sorting, format="memory", sparsity=None, name=""):
        sparse = sparsity is not None

        if format == "memory":
            folder = None
        elif format == "binary_folder":
            folder = self.cache_folder / f"test_{name}_sparse{sparse}_{format}"
        elif format == "zarr":
            folder = self.cache_folder / f"test_{name}_sparse{sparse}_{format}.zarr"
        if folder and folder.exists():
            shutil.rmtree(folder)

        sorting_analyzer = create_sorting_analyzer(
            sorting, recording, format=format, folder=folder, sparse=False, sparsity=sparsity
        )

        return sorting_analyzer

    def _prepare_sorting_analyzer(self, format, sparse, extension_class):
        # prepare a SortingAnalyzer object with depencies already computed
        sparsity_ = self.sparsity if sparse else None
        sorting_analyzer = self.get_sorting_analyzer(
            self.recording, self.sorting, format=format, sparsity=sparsity_, name=extension_class.extension_name
        )
        sorting_analyzer.compute("random_spikes", max_spikes_per_unit=50, seed=2205)

        for dependency_name in extension_class.depend_on:
            if "|" in dependency_name:
                dependency_name = dependency_name.split("|")[0]
            sorting_analyzer.compute(dependency_name)

        return sorting_analyzer

    def _check_one(self, sorting_analyzer, extension_class, params):
        """
        Take a prepared sorting analyzer object, compute the extension of interest
        with the passed parameters, and check the output is not empty, the extension
        exists and `select_units()` method works.
        """
        if extension_class.need_job_kwargs:
            job_kwargs = dict(n_jobs=2, chunk_duration="1s", progress_bar=True)
        else:
            job_kwargs = dict()

        ext = sorting_analyzer.compute(extension_class.extension_name, **params, **job_kwargs)
        assert len(ext.data) > 0
        main_data = ext.get_data()
        assert len(main_data) > 0

        ext = sorting_analyzer.get_extension(extension_class.extension_name)
        assert ext is not None

        some_unit_ids = sorting_analyzer.unit_ids[::2]
        sliced = sorting_analyzer.select_units(some_unit_ids, format="memory")
        assert np.array_equal(sliced.unit_ids, sorting_analyzer.unit_ids[::2])

    def run_extension_tests(self, extension_class, params):
        """
        Convenience function to perform all checks on the extension
        of interest with the passed parameters. Will perform tests
        for sparsity and format.
        """
        for sparse in (True, False):
            for format in ("memory", "binary_folder", "zarr"):
                print("sparse", sparse, format)
                sorting_analyzer = self._prepare_sorting_analyzer(format, sparse, extension_class)
                self._check_one(sorting_analyzer, extension_class, params)
