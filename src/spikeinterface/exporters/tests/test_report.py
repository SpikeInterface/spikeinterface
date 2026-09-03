from concurrent.futures import ProcessPoolExecutor
import hashlib
import platform
import shutil
import struct
from unittest import mock
import zlib

import pytest

from spikeinterface.core import create_sorting_analyzer, generate_ground_truth_recording, load_sorting_analyzer
from spikeinterface.exporters import export_report
from spikeinterface.exporters import report as report_module
from spikeinterface.exporters.tests.common import (
    make_sorting_analyzer,
    sorting_analyzer_dense_for_export,
    sorting_analyzer_sparse_for_export,
    sorting_analyzer_with_group_for_export,
)


def _unit_png_hashes(units_folder):
    return {
        png_path.stem: hashlib.sha256(png_path.read_bytes()).hexdigest()
        for png_path in sorted(units_folder.glob("*.png"))
    }


def _assert_png_file(figure_path):
    # Pillow would give a stronger check but is not a declared exporters/test-exporters dependency
    # (that gap is exactly what broke test-core collection in the bug this module regression-tests),
    # so this walks the chunk stream by hand instead of only checking the signature and trailer bytes.
    data = figure_path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    chunk_types = []
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        (crc_stored,) = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])
        crc_expected = zlib.crc32(chunk_type + chunk_data)
        assert crc_stored == crc_expected, f"corrupt PNG chunk {chunk_type!r} in {figure_path}"
        chunk_types.append(chunk_type)
        offset += 12 + length
    assert offset == len(data), f"trailing or truncated data after the last chunk in {figure_path}"
    assert chunk_types[0] == b"IHDR"
    assert chunk_types[-1] == b"IEND"


def _make_small_analyzer(folder, durations=(5.0,)):
    # UnitWaveformsWidget randomly subsamples down to 50 waveforms per unit when more are stored
    # (unit_waveforms.py, unseeded np.random.permutation), which makes rendering non-deterministic
    # across separate export_report calls regardless of parallelization. Keeping max_spikes_per_unit
    # at or below that widget default avoids the subsampling branch entirely, so serial and parallel
    # renders of the same analyzer can be compared byte-for-byte.
    recording, sorting = generate_ground_truth_recording(
        durations=list(durations),
        sampling_frequency=30_000.0,
        num_channels=8,
        num_units=4,
        generate_probe_kwargs=dict(
            num_columns=2, xpitch=20, ypitch=20, contact_shapes="circle", contact_shape_params={"radius": 6}
        ),
        generate_sorting_kwargs=dict(firing_rates=8.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=5.0, strategy="on_the_fly"),
        seed=2205,
    )
    sorting_analyzer = create_sorting_analyzer(
        sorting=sorting, recording=recording, format="binary_folder", folder=folder, sparse=True
    )
    sorting_analyzer.compute("random_spikes", max_spikes_per_unit=40)
    sorting_analyzer.compute("waveforms", n_jobs=1, progress_bar=False)
    sorting_analyzer.compute("templates")
    sorting_analyzer.compute("unit_locations")
    sorting_analyzer.compute("correlograms")
    sorting_analyzer.compute("spike_amplitudes", n_jobs=1, progress_bar=False)
    return sorting_analyzer


def test_export_report(sorting_analyzer_sparse_for_export, create_cache_folder):
    cache_folder = create_cache_folder
    report_folder = cache_folder / "report"
    if report_folder.exists():
        shutil.rmtree(report_folder)

    sorting_analyzer = sorting_analyzer_sparse_for_export

    job_kwargs = dict(n_jobs=1, chunk_size=30000, progress_bar=True)
    export_report(sorting_analyzer, report_folder, force_computation=True, **job_kwargs)


def test_export_report_parallel_spawn_binary_folder(sorting_analyzer_sparse_for_export, create_cache_folder):
    cache_folder = create_cache_folder
    analyzer_folder = cache_folder / "analyzer_binary_folder"
    report_folder = cache_folder / "parallel_report_binary_folder"
    sorting_analyzer = sorting_analyzer_sparse_for_export.save_as(format="binary_folder", folder=analyzer_folder)

    with mock.patch.object(report_module, "ProcessPoolExecutor", side_effect=ProcessPoolExecutor) as executor:
        export_report(
            sorting_analyzer,
            report_folder,
            n_jobs=2,
            mp_context="spawn",
            max_threads_per_worker=1,
            progress_bar=False,
        )

    executor.assert_called_once()

    unit_figures = sorted((report_folder / "units").glob("*.png"))
    assert len(unit_figures) == sorting_analyzer.get_num_units()
    for figure_path in unit_figures:
        _assert_png_file(figure_path)


def test_export_report_parallel_zarr_falls_back_to_serial(sorting_analyzer_dense_for_export, create_cache_folder):
    analyzer_folder = create_cache_folder / "analyzer_zarr"
    report_folder = create_cache_folder / "report_zarr"
    sorting_analyzer = sorting_analyzer_dense_for_export.save_as(format="zarr", folder=analyzer_folder)

    with mock.patch.object(report_module, "ProcessPoolExecutor", side_effect=ProcessPoolExecutor) as executor:
        with pytest.warns(UserWarning, match="Zarr SortingAnalyzer would load extension arrays once per worker"):
            export_report(
                sorting_analyzer,
                report_folder,
                n_jobs=2,
                mp_context="spawn",
                max_threads_per_worker=1,
                progress_bar=False,
            )

    executor.assert_not_called()
    unit_figures = sorted((report_folder / "units").glob("*.png"))
    assert len(unit_figures) == sorting_analyzer.get_num_units()
    for figure_path in unit_figures:
        _assert_png_file(figure_path)


@pytest.mark.parametrize(
    "mp_context",
    (
        "spawn",
        pytest.param(
            "fork",
            marks=pytest.mark.skipif(
                platform.system() != "Linux", reason="the export_report 'fork' path is only supported on Linux"
            ),
        ),
    ),
)
def test_export_report_parallel_matches_serial_binary_folder(mp_context, create_cache_folder):
    cache_folder = create_cache_folder
    analyzer_folder = cache_folder / f"small_analyzer_{mp_context}"
    serial_folder = cache_folder / f"small_serial_report_{mp_context}"
    report_folder = cache_folder / f"small_parallel_report_{mp_context}"
    sorting_analyzer = _make_small_analyzer(analyzer_folder)

    export_report(sorting_analyzer, serial_folder, n_jobs=1, progress_bar=False)
    serial_hashes = _unit_png_hashes(serial_folder / "units")
    assert len(serial_hashes) == sorting_analyzer.get_num_units()

    with mock.patch.object(report_module, "ProcessPoolExecutor", side_effect=ProcessPoolExecutor) as executor:
        export_report(
            sorting_analyzer,
            report_folder,
            n_jobs=2,
            mp_context=mp_context,
            max_threads_per_worker=1,
            progress_bar=False,
        )

    executor.assert_called_once()
    # the worker reloads the analyzer from disk (binary_folder): its rendered output must match serial exactly
    assert _unit_png_hashes(report_folder / "units") == serial_hashes


def test_export_report_parallel_matches_serial_two_segments(create_cache_folder):
    # each segment contributes its own spikes/waveforms; a worker reloading a multi-segment analyzer
    # from disk must see all of them, not just segment 0, for its render to match the serial one.
    cache_folder = create_cache_folder
    analyzer_folder = cache_folder / "small_analyzer_two_segments"
    serial_folder = cache_folder / "small_serial_report_two_segments"
    report_folder = cache_folder / "small_parallel_report_two_segments"
    sorting_analyzer = _make_small_analyzer(analyzer_folder, durations=(5.0, 5.0))
    assert sorting_analyzer.get_num_segments() == 2

    export_report(sorting_analyzer, serial_folder, n_jobs=1, progress_bar=False)
    serial_hashes = _unit_png_hashes(serial_folder / "units")
    assert len(serial_hashes) == sorting_analyzer.get_num_units()

    with mock.patch.object(report_module, "ProcessPoolExecutor", side_effect=ProcessPoolExecutor) as executor:
        export_report(
            sorting_analyzer,
            report_folder,
            n_jobs=2,
            mp_context="spawn",
            max_threads_per_worker=1,
            progress_bar=False,
        )

    executor.assert_called_once()
    assert _unit_png_hashes(report_folder / "units") == serial_hashes


@pytest.mark.skipif(platform.system() != "Linux", reason="the export_report 'fork' path is only supported on Linux")
def test_export_report_parallel_fork_memory(sorting_analyzer_sparse_for_export, create_cache_folder):
    # in-memory analyzers take the fork fast path: the analyzer object itself (not a folder) is
    # sent to each worker, instead of being reloaded from disk. This is the one parallel path with
    # no disk round-trip, so it needs its own coverage distinct from the spawn/binary_folder tests.
    report_folder = create_cache_folder / "parallel_report_memory_fork"

    with mock.patch.object(report_module, "ProcessPoolExecutor", side_effect=ProcessPoolExecutor) as executor:
        export_report(
            sorting_analyzer_sparse_for_export,
            report_folder,
            n_jobs=2,
            mp_context="fork",
            max_threads_per_worker=1,
            progress_bar=False,
        )

    executor.assert_called_once()
    unit_figures = sorted((report_folder / "units").glob("*.png"))
    assert len(unit_figures) == sorting_analyzer_sparse_for_export.get_num_units()
    for figure_path in unit_figures:
        _assert_png_file(figure_path)


def test_init_export_report_worker_uses_known_format_and_backend_options(create_cache_folder):
    # the worker must be told the SortingAnalyzer's actual format and backend_options explicitly:
    # load_sorting_analyzer's format="auto" default only looks at the folder name (a binary_folder
    # ending in ".zarr" would be misdetected), and omitting backend_options drops any storage
    # credentials the original analyzer was opened with.
    misleading_folder = create_cache_folder / "worker_format_analyzer.zarr"
    sorting_analyzer = _make_small_analyzer(misleading_folder)

    backend_options = {"storage_options": {}}
    with mock.patch("spikeinterface.core.load_sorting_analyzer", side_effect=load_sorting_analyzer) as loader:
        report_module._init_export_report_worker(
            str(misleading_folder),
            "binary_folder",
            backend_options,
            create_cache_folder / "worker_format_units",
            "png",
            None,
        )

    loader.assert_called_once_with(
        str(misleading_folder),
        format="binary_folder",
        backend_options=backend_options,
        lazy=True,
        load_extensions=False,
    )
    loaded_analyzer, *_ = report_module._export_report_worker_context
    assert loaded_analyzer.format == "binary_folder"
    assert loaded_analyzer.get_num_units() == sorting_analyzer.get_num_units()


def test_export_report_parallel_spawn_memory_fallback(sorting_analyzer_sparse_for_export, create_cache_folder):
    report_folder = create_cache_folder / "parallel_report_memory"
    with pytest.warns(UserWarning, match="Save the SortingAnalyzer as `binary_folder`"):
        export_report(
            sorting_analyzer_sparse_for_export,
            report_folder,
            n_jobs=2,
            mp_context="spawn",
            progress_bar=False,
        )


def test_export_report_parallel_falls_back_for_read_only_analyzer(create_cache_folder):
    # a worker reloading a read-only SortingAnalyzer folder cannot be told to just not save (there is
    # nothing to save, it only reads), but export_report must still not attempt to reload it into
    # multiple worker processes as if it were an ordinary writable folder: is_read_only() must be
    # checked before deciding to parallelize, not discovered as a write failure inside a worker.
    analyzer_folder = create_cache_folder / "analyzer_read_only"
    report_folder = create_cache_folder / "parallel_report_read_only"
    sorting_analyzer = _make_small_analyzer(analyzer_folder)

    with mock.patch.object(sorting_analyzer, "is_read_only", return_value=True):
        with mock.patch.object(report_module, "ProcessPoolExecutor", side_effect=ProcessPoolExecutor) as executor:
            with pytest.warns(UserWarning, match="writable `binary_folder` SortingAnalyzer"):
                export_report(
                    sorting_analyzer,
                    report_folder,
                    n_jobs=2,
                    mp_context="spawn",
                    max_threads_per_worker=1,
                    progress_bar=False,
                )
        executor.assert_not_called()

    unit_figures = sorted((report_folder / "units").glob("*.png"))
    assert len(unit_figures) == sorting_analyzer.get_num_units()


def test_export_report_parallel_lazy_analyzer_falls_back_to_serial(
    sorting_analyzer_sparse_for_export, create_cache_folder
):
    # a lazy analyzer forces save=False on any extension it computes (SortingAnalyzer.compute());
    # unit_locations is computed unconditionally by export_report, so a lazy analyzer without it
    # already on disk must not take the parallel path, since the worker reloads from disk and would
    # not see it (see SortingAnalyzer.compute and AnalyzerExtension.check_extensions)
    from spikeinterface.core import load_sorting_analyzer

    analyzer_folder = create_cache_folder / "analyzer_lazy_fallback"
    report_folder = create_cache_folder / "parallel_report_lazy_fallback"
    saved = sorting_analyzer_sparse_for_export.save_as(format="binary_folder", folder=analyzer_folder)
    saved.delete_extension("unit_locations")

    lazy_analyzer = load_sorting_analyzer(analyzer_folder, lazy=True)
    assert not lazy_analyzer.has_extension("unit_locations")

    with mock.patch.object(report_module, "ProcessPoolExecutor", side_effect=ProcessPoolExecutor) as executor:
        with pytest.warns(UserWarning, match="computed in memory only"):
            export_report(
                lazy_analyzer,
                report_folder,
                n_jobs=2,
                mp_context="spawn",
                max_threads_per_worker=1,
                progress_bar=False,
            )

    executor.assert_not_called()
    unit_figures = sorted((report_folder / "units").glob("*.png"))
    assert len(unit_figures) == lazy_analyzer.get_num_units()


def test_export_report_parallel_falls_back_when_saved_extension_recomputed_with_save_false(create_cache_folder):
    # SortingAnalyzer.compute(..., save=False) keeps the on-disk extension of the same name untouched
    # but recomputes it in memory with different params (this is its documented use: "try some
    # parameters without changing an already saved extension"). A name-only check would consider
    # "unit_locations" saved and take the parallel path, so the main process (global figures) would use
    # the fresh in-memory locations while the workers reload the stale on-disk ones.
    analyzer_folder = create_cache_folder / "analyzer_stale_recompute"
    report_folder = create_cache_folder / "parallel_report_stale_recompute"
    sorting_analyzer = _make_small_analyzer(analyzer_folder)
    assert sorting_analyzer.get_extension("unit_locations").params.get("method") == "monopolar_triangulation"

    sorting_analyzer.compute("unit_locations", method="center_of_mass", save=False)
    assert sorting_analyzer.get_extension("unit_locations").params.get("method") == "center_of_mass"

    with mock.patch.object(report_module, "ProcessPoolExecutor", side_effect=ProcessPoolExecutor) as executor:
        with pytest.warns(UserWarning, match="different parameters than on disk"):
            export_report(
                sorting_analyzer,
                report_folder,
                n_jobs=2,
                mp_context="spawn",
                max_threads_per_worker=1,
                progress_bar=False,
            )

    executor.assert_not_called()
    unit_figures = sorted((report_folder / "units").glob("*.png"))
    assert len(unit_figures) == sorting_analyzer.get_num_units()


if __name__ == "__main__":
    sorting_analyzer = make_sorting_analyzer(sparse=True)
    test_export_report(sorting_analyzer)
