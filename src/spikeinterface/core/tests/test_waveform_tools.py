import pytest
from pathlib import Path
import shutil
import platform

import numpy as np

from spikeinterface.core import generate_recording, generate_sorting, generate_ground_truth_recording
from spikeinterface.core.waveform_tools import (
    extract_waveforms_to_buffers,
    extract_waveforms_to_single_buffer,
    split_waveforms_by_units,
    estimate_templates,
    estimate_templates_with_accumulator,
)


def _check_all_wf_equal(list_wfs_arrays):
    wfs_arrays0 = list_wfs_arrays[0]
    for i, wfs_arrays in enumerate(list_wfs_arrays):
        for unit_id in wfs_arrays.keys():
            assert np.array_equal(wfs_arrays[unit_id], wfs_arrays0[unit_id])


def get_dataset():
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0, 40.0],
        sampling_frequency=30000.0,
        num_channels=4,
        num_units=7,
        generate_sorting_kwargs=dict(firing_rates=5.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=1.0, strategy="tile_pregenerated"),
        seed=2205,
    )
    return recording, sorting


def test_waveform_tools(create_cache_folder):
    cache_folder = create_cache_folder
    # durations = [30, 40]
    # sampling_frequency = 30000.0

    # # 2 segments
    # num_channels = 2
    # recording = generate_recording(
    #     num_channels=num_channels, durations=durations, sampling_frequency=sampling_frequency
    # )
    # recording.annotate(is_filtered=True)
    # num_units = 15
    # sorting = generate_sorting(num_units=num_units, sampling_frequency=sampling_frequency, durations=durations)

    # test with dump !!!!
    # recording = recording.save()
    # sorting = sorting.save()

    recording, sorting = get_dataset()
    sampling_frequency = recording.sampling_frequency

    nbefore = int(3.0 * sampling_frequency / 1000.0)
    nafter = int(4.0 * sampling_frequency / 1000.0)

    dtype = recording.get_dtype()
    # return_in_uV = False

    spikes = sorting.to_spike_vector()

    unit_ids = sorting.unit_ids

    some_job_kwargs = [
        {"n_jobs": 1, "chunk_size": 3000, "progress_bar": True},
        {"n_jobs": 2, "chunk_size": 3000, "progress_bar": True},
    ]
    some_modes = [
        {"mode": "memmap"},
        {"mode": "shared_memory"},
    ]
    # if platform.system() != "Windows":
    #     # shared memory on windows is buggy...
    #     some_modes.append(
    #         {
    #             "mode": "shared_memory",
    #         }
    #     )

    some_sparsity = [
        dict(sparsity_mask=None),
        dict(sparsity_mask=np.random.randint(0, 2, size=(unit_ids.size, recording.channel_ids.size), dtype="bool")),
    ]

    # memmap mode
    list_wfs_dense = []
    list_wfs_sparse = []
    for j, job_kwargs in enumerate(some_job_kwargs):
        for k, mode_kwargs in enumerate(some_modes):
            for l, sparsity_kwargs in enumerate(some_sparsity):
                # print()
                # print(job_kwargs, mode_kwargs, 'sparse=', sparsity_kwargs['sparsity_mask'] is None)

                if mode_kwargs["mode"] == "memmap":
                    wf_folder = cache_folder / f"test_waveform_tools_{j}_{k}_{l}"
                    if wf_folder.is_dir():
                        shutil.rmtree(wf_folder)
                    wf_folder.mkdir(parents=True)
                    wf_file_path = wf_folder / "waveforms_all_units.npy"

                mode_kwargs_ = dict(**mode_kwargs)
                if mode_kwargs["mode"] == "memmap":
                    mode_kwargs_["folder"] = wf_folder

                wfs_arrays = extract_waveforms_to_buffers(
                    recording,
                    spikes,
                    unit_ids,
                    nbefore,
                    nafter,
                    return_in_uV=False,
                    dtype=dtype,
                    copy=True,
                    **sparsity_kwargs,
                    **mode_kwargs_,
                    **job_kwargs,
                )
                for unit_ind, unit_id in enumerate(unit_ids):
                    wf = wfs_arrays[unit_id]
                    assert wf.shape[0] == np.sum(spikes["unit_index"] == unit_ind)

                if sparsity_kwargs["sparsity_mask"] is None:
                    list_wfs_dense.append(wfs_arrays)
                else:
                    list_wfs_sparse.append(wfs_arrays)

                mode_kwargs_ = dict(**mode_kwargs)
                if mode_kwargs["mode"] == "memmap":
                    mode_kwargs_["file_path"] = wf_file_path

                all_waveforms = extract_waveforms_to_single_buffer(
                    recording,
                    spikes,
                    unit_ids,
                    nbefore,
                    nafter,
                    return_in_uV=False,
                    dtype=dtype,
                    copy=True,
                    **sparsity_kwargs,
                    **mode_kwargs_,
                    **job_kwargs,
                )
                wfs_arrays = split_waveforms_by_units(
                    unit_ids, spikes, all_waveforms, sparsity_mask=sparsity_kwargs["sparsity_mask"]
                )
                if sparsity_kwargs["sparsity_mask"] is None:
                    list_wfs_dense.append(wfs_arrays)
                else:
                    list_wfs_sparse.append(wfs_arrays)

    _check_all_wf_equal(list_wfs_dense)
    _check_all_wf_equal(list_wfs_sparse)


def test_estimate_templates_with_accumulator():
    recording, sorting = get_dataset()

    ms_before = 1.0
    ms_after = 1.5

    nbefore = int(ms_before * recording.sampling_frequency / 1000.0)
    nafter = int(ms_after * recording.sampling_frequency / 1000.0)

    spikes = sorting.to_spike_vector()
    # take one spikes every 10
    spikes = spikes[::10]

    job_kwargs = dict(n_jobs=2, progress_bar=True, chunk_duration="1s")

    # here we compare the result with the same mechanism with with several worker pool size
    # this means that that acumulator are splitted and then agglomerated back
    # this should lead to very small diff
    # n_jobs=1 is done in loop
    templates_by_worker = []

    if platform.system() == "Linux":
        engine_loop = ["thread", "process"]
    else:
        engine_loop = ["thread"]

    for pool_engine in engine_loop:
        for n_jobs in (1, 2, 8):
            job_kwargs = dict(pool_engine=pool_engine, n_jobs=n_jobs, progress_bar=True, chunk_duration="1s")
            templates = estimate_templates_with_accumulator(
                recording, spikes, sorting.unit_ids, nbefore, nafter, return_in_uV=True, **job_kwargs
            )
            assert templates.shape[0] == sorting.unit_ids.size
            assert templates.shape[1] == nbefore + nafter
            assert templates.shape[2] == recording.get_num_channels()
            assert np.any(templates != 0)

            templates_by_worker.append(templates)
            if len(templates_by_worker) > 1:
                templates_loop = templates_by_worker[0]
                np.testing.assert_almost_equal(templates, templates_loop, decimal=4)

                # import matplotlib.pyplot as plt
                # fig, axs = plt.subplots(nrows=2, sharex=True)
                # for unit_index, unit_id in enumerate(sorting.unit_ids):
                #     ax = axs[0]
                #     ax.set_title(f"{pool_engine} {n_jobs}")
                #     ax.plot(templates[unit_index, :, :].T.flatten())
                #     ax.plot(templates_loop[unit_index, :, :].T.flatten(), color="k", ls="--")
                #     ax = axs[1]
                #     ax.plot((templates - templates_loop)[unit_index, :, :].T.flatten(), color="k", ls="--")
                # plt.show()


def test_estimate_templates():
    recording, sorting = get_dataset()

    ms_before = 1.0
    ms_after = 1.5

    nbefore = int(ms_before * recording.sampling_frequency / 1000.0)
    nafter = int(ms_after * recording.sampling_frequency / 1000.0)

    spikes = sorting.to_spike_vector()
    # take one spikes every 10
    spikes = spikes[::10]

    job_kwargs = dict(n_jobs=2, progress_bar=True, chunk_duration="1s")

    # mask with differents sparsity
    sparsity_mask = np.ones((sorting.unit_ids.size, recording.channel_ids.size), dtype=bool)
    sparsity_mask[:4, : recording.channel_ids.size // 2 - 1] = False
    sparsity_mask[4:, recording.channel_ids.size // 2 :] = False

    for operator in ("average", "median"):
        templates_array = estimate_templates(
            recording, spikes, sorting.unit_ids, nbefore, nafter, operator=operator, return_in_uV=True, **job_kwargs
        )
        # print(templates.shape)
        assert templates_array.shape[0] == sorting.unit_ids.size
        assert templates_array.shape[1] == nbefore + nafter
        assert templates_array.shape[2] == recording.get_num_channels()

        assert np.any(templates_array != 0)

        sparse_templates_array = estimate_templates(
            recording,
            spikes,
            sorting.unit_ids,
            nbefore,
            nafter,
            operator=operator,
            return_in_uV=True,
            sparsity_mask=sparsity_mask,
            **job_kwargs,
        )
        n_chan = np.max(np.sum(sparsity_mask, axis=1))
        assert n_chan == sparse_templates_array.shape[2]
        assert np.any(sparse_templates_array == 0)

    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots()
    #     for unit_index, unit_id in enumerate(sorting.unit_ids):
    #         ax.plot(templates[unit_index, :, :].T.flatten())

    # plt.show()


if __name__ == "__main__":
    cache_folder = Path(__file__).resolve().parents[4] / "cache_folder" / "core"
    test_waveform_tools(cache_folder)
    # test_estimate_templates_with_accumulator()
    # test_estimate_templates()
