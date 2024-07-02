"""
This module contains low-level functions to extract snippets of traces (aka "spike waveforms").

This is internally used by SortingAnalyzer, but can also be used as a sorting component.

It is a 2-step approach:
  1. allocate buffers (shared file or memory)
  2. extract and distribute snippets into buffers (optionally in parallel)

"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import multiprocessing

from spikeinterface.core.baserecording import BaseRecording

from .baserecording import BaseRecording
from .job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc
from .core_tools import make_shared_array
from .job_tools import fix_job_kwargs


def extract_waveforms_to_buffers(
    recording,
    spikes,
    unit_ids,
    nbefore,
    nafter,
    mode="memmap",
    return_scaled=False,
    folder=None,
    dtype=None,
    sparsity_mask=None,
    copy=False,
    **job_kwargs,
):
    """
    Allocate buffers (memmap or or shared memory) and then distribute every waveform into theses buffers.

    Same as calling allocate_waveforms_buffers() and then distribute_waveforms_to_buffers().

    Important note: for the "shared_memory" mode arrays_info contains reference to
    the shared memmory buffer, this variable must be reference as long as arrays as used.
    And this variable is also returned.
    To avoid this a copy to non shared memmory can be perform at the end.

    Parameters
    ----------
    recording: recording
        The recording object
    spikes: 1d numpy array with several fields
        Spikes handled as a unique vector.
        This vector can be obtained with: `spikes = Sorting.to_spike_vector()`
    unit_ids: list ot numpy
        List of unit_ids
    nbefore: int
        N samples before spike
    nafter: int
        N samples after spike
    mode: "memmap" | "shared_memory", default: "memmap"
        The mode to use for the buffer
    return_scaled: bool, default: False
        Scale traces before exporting to buffer or not
    folder: str or path or None, default: None
        In case of memmap mode, folder to save npy files
    dtype: numpy.dtype, default: None
        dtype for waveforms buffer
    sparsity_mask: None or array of bool, default: None
        If not None shape must be must be (len(unit_ids), len(channel_ids))
    copy: bool, default: False
        If True, the output shared memory object is copied to a numpy standard array.
        If copy=False then arrays_info is also return. Please keep in mind that arrays_info
        need to be referenced as long as waveforms_by_units will be used otherwise it will be very hard to debug.
        Also when copy=False the SharedMemory will need to be unlink manually
    {}

    Returns
    -------
    waveforms_by_units: dict of arrays
        Arrays for all units (memmap or shared_memmep)

    arrays_info: dict of info
        Optionally return in case of shared_memory if copy=False.
        Dictionary to "construct" array in workers process (memmap file or sharemem info)
    """
    job_kwargs = fix_job_kwargs(job_kwargs)

    if dtype is None:
        if return_scaled:
            dtype = recording.get_dtype()
        else:
            dtype = "float32"
    dtype = np.dtype(dtype)

    waveforms_by_units, arrays_info = allocate_waveforms_buffers(
        recording, spikes, unit_ids, nbefore, nafter, mode=mode, folder=folder, dtype=dtype, sparsity_mask=sparsity_mask
    )

    distribute_waveforms_to_buffers(
        recording,
        spikes,
        unit_ids,
        arrays_info,
        nbefore,
        nafter,
        return_scaled,
        mode=mode,
        sparsity_mask=sparsity_mask,
        **job_kwargs,
    )

    if mode == "memmap":
        return waveforms_by_units
    elif mode == "shared_memory":
        if copy:
            waveforms_by_units = {unit_id: arr.copy() for unit_id, arr in waveforms_by_units.items()}
            # release all sharedmem buffer
            for unit_id in unit_ids:
                shm = arrays_info[unit_id][0]
                if shm is not None:
                    # empty array have None
                    shm.unlink()
            return waveforms_by_units
        else:
            return waveforms_by_units, arrays_info


extract_waveforms_to_buffers.__doc__ = extract_waveforms_to_buffers.__doc__.format(_shared_job_kwargs_doc)


def allocate_waveforms_buffers(
    recording, spikes, unit_ids, nbefore, nafter, mode="memmap", folder=None, dtype=None, sparsity_mask=None
):
    """
    Allocate memmap or shared memory buffers before snippet extraction.

    Important note: for the shared memory mode arrays_info contains reference to
    the shared memmory buffer, this variable must be reference as long as arrays as used.

    Parameters
    ----------
    recording: recording
        The recording object
    spikes: 1d numpy array with several fields
        Spikes handled as a unique vector.
        This vector can be obtained with: `spikes = Sorting.to_spike_vector()`
    unit_ids: list ot numpy
        List of unit_ids
    nbefore: int
        N samples before spike
    nafter: int
        N samples after spike
    mode: "memmap" | "shared_memory", default: "memmap"
        Mode to use
    folder: str or path
        In case of memmap mode, folder to save npy files
    dtype: numpy.dtype
        dtype for waveforms buffer
    sparsity_mask: None or array of bool
        If not None shape must be must be (len(unit_ids), len(channel_ids)

    Returns
    -------
    waveforms_by_units: dict of arrays
        Arrays for all units (memmap or shared_memmep
    arrays_info: dict of info
        Dictionary to "construct" array in workers process (memmap file or sharemem)
    """

    nsamples = nbefore + nafter

    dtype = np.dtype(dtype)
    if mode == "shared_memory":
        assert folder is None
    else:
        folder = Path(folder)

    # prepare buffers
    waveforms_by_units = {}
    arrays_info = {}
    for unit_ind, unit_id in enumerate(unit_ids):
        n_spikes = np.sum(spikes["unit_index"] == unit_ind)
        if sparsity_mask is None:
            num_chans = recording.get_num_channels()
        else:
            num_chans = np.sum(sparsity_mask[unit_ind, :])
        shape = (n_spikes, nsamples, num_chans)

        if mode == "memmap":
            filename = str(folder / f"waveforms_{unit_id}.npy")
            arr = np.lib.format.open_memmap(filename, mode="w+", dtype=dtype, shape=shape)
            waveforms_by_units[unit_id] = arr
            arrays_info[unit_id] = filename
        elif mode == "shared_memory":
            if n_spikes == 0 or num_chans == 0:
                arr = np.zeros(shape, dtype=dtype)
                shm = None
                shm_name = None
            else:
                arr, shm = make_shared_array(shape, dtype)
                shm_name = shm.name
            waveforms_by_units[unit_id] = arr
            arrays_info[unit_id] = (shm, shm_name, dtype.str, shape)
        else:
            raise ValueError("allocate_waveforms_buffers bad mode")

    return waveforms_by_units, arrays_info


def distribute_waveforms_to_buffers(
    recording,
    spikes,
    unit_ids,
    arrays_info,
    nbefore,
    nafter,
    return_scaled,
    mode="memmap",
    sparsity_mask=None,
    job_name=None,
    verbose=False,
    **job_kwargs,
):
    """
    Distribute snippets of traces into corresponding buffers.

    Buffers must be pre-allocated with the `allocate_waveforms_buffers()` function.

    Important note, for "shared_memory" mode arrays_info contain reference to
    the shared memmory buffer, this variable must be reference as long as arrays as used.

    Parameters
    ----------
    recording: recording
        The recording object
    spikes: 1d numpy array with several field
        Spikes handled as a unique vector.
        This vector can be spikes = Sorting.to_spike_vector()
    unit_ids: list ot numpy
        List of unit_ids
    arrays_info: dict
        Dictionary to "construct" array in workers process (memmap file or sharemem)
    nbefore: int
        N samples before spike
    nafter: int
        N samples after spike
    return_scaled: bool
        Scale traces before exporting to buffer or not.
    mode: "memmap" | "shared_memory", default: "memmap"
        Mode to use
    sparsity_mask: None or array of bool
        If not None shape must be must be (len(unit_ids), len(channel_ids)

    {}

    """
    job_kwargs = fix_job_kwargs(job_kwargs)

    inds_by_unit = {}
    for unit_ind, unit_id in enumerate(unit_ids):
        (inds,) = np.nonzero(spikes["unit_index"] == unit_ind)
        inds_by_unit[unit_id] = inds

    # and run
    func = _worker_distribute_buffers
    init_func = _init_worker_distribute_buffers

    init_args = (
        recording,
        unit_ids,
        spikes,
        arrays_info,
        nbefore,
        nafter,
        return_scaled,
        inds_by_unit,
        mode,
        sparsity_mask,
    )
    if job_name is None:
        job_name = f"extract waveforms {mode} multi buffer"
    processor = ChunkRecordingExecutor(
        recording, func, init_func, init_args, job_name=job_name, verbose=verbose, **job_kwargs
    )
    processor.run()


distribute_waveforms_to_buffers.__doc__ = distribute_waveforms_to_buffers.__doc__.format(_shared_job_kwargs_doc)


# used by ChunkRecordingExecutor
def _init_worker_distribute_buffers(
    recording, unit_ids, spikes, arrays_info, nbefore, nafter, return_scaled, inds_by_unit, mode, sparsity_mask
):
    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor

        recording = load_extractor(recording)
    worker_ctx["recording"] = recording

    if mode == "memmap":
        # in memmap mode we have the "too many open file" problem with linux
        # memmap file will be open on demand and not globally per worker
        worker_ctx["arrays_info"] = arrays_info
    elif mode == "shared_memory":
        from multiprocessing.shared_memory import SharedMemory

        waveforms_by_units = {}
        shms = {}
        for unit_id, (shm, shm_name, dtype, shape) in arrays_info.items():
            if shm_name is None:
                arr = np.zeros(shape=shape, dtype=dtype)
            else:
                shm = SharedMemory(shm_name)
                arr = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
            waveforms_by_units[unit_id] = arr
            # we need a reference to all sham otherwise we get segment fault!!!
            shms[unit_id] = shm
        worker_ctx["shms"] = shms
        worker_ctx["waveforms_by_units"] = waveforms_by_units

    worker_ctx["unit_ids"] = unit_ids
    worker_ctx["spikes"] = spikes

    worker_ctx["nbefore"] = nbefore
    worker_ctx["nafter"] = nafter
    worker_ctx["return_scaled"] = return_scaled
    worker_ctx["inds_by_unit"] = inds_by_unit
    worker_ctx["sparsity_mask"] = sparsity_mask
    worker_ctx["mode"] = mode

    return worker_ctx


# used by ChunkRecordingExecutor
def _worker_distribute_buffers(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx["recording"]
    unit_ids = worker_ctx["unit_ids"]
    spikes = worker_ctx["spikes"]
    nbefore = worker_ctx["nbefore"]
    nafter = worker_ctx["nafter"]
    return_scaled = worker_ctx["return_scaled"]
    inds_by_unit = worker_ctx["inds_by_unit"]
    sparsity_mask = worker_ctx["sparsity_mask"]

    seg_size = recording.get_num_samples(segment_index=segment_index)

    # take only spikes with the correct segment_index
    # this is a slice so no copy!!
    s0, s1 = np.searchsorted(spikes["segment_index"], [segment_index, segment_index + 1])
    in_seg_spikes = spikes[s0:s1]

    # take only spikes in range [start_frame, end_frame]
    # this is a slice so no copy!!
    # the border of segment are protected by nbefore on left an nafter on the right
    i0, i1 = np.searchsorted(
        in_seg_spikes["sample_index"], [max(start_frame, nbefore), min(end_frame, seg_size - nafter)]
    )

    # slice in absolut in spikes vector
    l0 = i0 + s0
    l1 = i1 + s0

    if l1 > l0:
        start = spikes[l0]["sample_index"] - nbefore
        end = spikes[l1 - 1]["sample_index"] + nafter

        # load trace in memory
        traces = recording.get_traces(
            start_frame=start, end_frame=end, segment_index=segment_index, return_scaled=return_scaled
        )

        for unit_ind, unit_id in enumerate(unit_ids):
            # find pos
            inds = inds_by_unit[unit_id]
            (in_chunk_pos,) = np.nonzero((inds >= l0) & (inds < l1))
            if in_chunk_pos.size == 0:
                continue

            if worker_ctx["mode"] == "memmap":
                # open file in demand (and also autoclose it after)
                filename = worker_ctx["arrays_info"][unit_id]
                wfs = np.load(str(filename), mmap_mode="r+")
            elif worker_ctx["mode"] == "shared_memory":
                wfs = worker_ctx["waveforms_by_units"][unit_id]

            for pos in in_chunk_pos:
                sample_index = spikes[inds[pos]]["sample_index"]
                wf = traces[sample_index - start - nbefore : sample_index - start + nafter, :]

                if sparsity_mask is None:
                    wfs[pos, :, :] = wf
                else:
                    wfs[pos, :, :] = wf[:, sparsity_mask[unit_ind]]


def extract_waveforms_to_single_buffer(
    recording,
    spikes,
    unit_ids,
    nbefore,
    nafter,
    mode="memmap",
    return_scaled=False,
    file_path=None,
    dtype=None,
    sparsity_mask=None,
    copy=True,
    job_name=None,
    verbose=False,
    **job_kwargs,
):
    """
    Allocate a single buffer (memmap or or shared memory) and then distribute every waveform into it.

    Contrary to extract_waveforms_to_buffers() all waveforms are extracted in the same buffer, so the spike vector is
    needed to recover waveforms unit by unit. Importantly in case of sparsity, the channels are not aligned across
    units.

    Note: spikes near borders (nbefore/nafter) are not extracted and 0 are put the output buffer.
    This ensures that spikes.shape[0] == all_waveforms.shape[0].

    Important note: for the "shared_memory" mode wf_array_info contains reference to
    the shared memmory buffer, this variable must be referenced as long as arrays is used.
    This variable must also unlink() when the array is de-referenced.
    To avoid this complicated behavior, default: (copy=True) the shared memmory buffer is copied into a standard
    numpy array.


    Parameters
    ----------
    recording: recording
        The recording object
    spikes: 1d numpy array with several fields
        Spikes handled as a unique vector.
        This vector can be obtained with: `spikes = Sorting.to_spike_vector()`
    unit_ids: list ot numpy
        List of unit_ids
    nbefore: int
        N samples before spike
    nafter: int
        N samples after spike
    mode: "memmap" | "shared_memory", default: "memmap"
        The mode to use for the buffer
    return_scaled: bool, default: False
        Scale traces before exporting to buffer or not
    file_path: str or path or None, default: None
        In case of memmap mode, file to save npy file
    dtype: numpy.dtype, default: None
        dtype for waveforms buffer
    sparsity_mask: None or array of bool, default: None
        If not None shape must be must be (len(unit_ids), len(channel_ids))
    copy: bool, default: False
        If True, the output shared memory object is copied to a numpy standard array and no reference
        to the internal shared memory object is kept.
        If copy=False then the shared memory object is also returned. Please keep in mind that the shared memory object
        need to be referenced as long as all_waveforms will be used otherwise it might produce segmentation
        faults which are hard to debug.
        Also when copy=False the SharedMemory will need to be unlink manually if proper cleanup of resources is desired.

    {}

    Returns
    -------
    all_waveforms: numpy array
        Single array with shape (nump_spikes, num_samples, num_channels)

    wf_array_info: dict of info
        Optionally return in case of shared_memory if copy=False.
        Dictionary to "construct" array in workers process (memmap file or sharemem info)
    """
    nsamples = nbefore + nafter

    dtype = np.dtype(dtype)
    if mode == "shared_memory":
        assert file_path is None
    else:
        file_path = Path(file_path)

    num_spikes = spikes.size
    if sparsity_mask is None:
        num_chans = recording.get_num_channels()
    else:
        num_chans = int(max(np.sum(sparsity_mask, axis=1)))  # This is a numpy scalar, so we cast to int
    shape = (num_spikes, nsamples, num_chans)

    if mode == "memmap":
        all_waveforms = np.lib.format.open_memmap(file_path, mode="w+", dtype=dtype, shape=shape)
        # wf_array_info = str(file_path)
        wf_array_info = dict(filename=str(file_path))
    elif mode == "shared_memory":
        if num_spikes == 0 or num_chans == 0:
            all_waveforms = np.zeros(shape, dtype=dtype)
            shm = None
            shm_name = None
        else:
            all_waveforms, shm = make_shared_array(shape, dtype)
            shm_name = shm.name
        # wf_array_info = (shm, shm_name, dtype.str, shape)
        wf_array_info = dict(shm=shm, shm_name=shm_name, dtype=dtype.str, shape=shape)
    else:
        raise ValueError("allocate_waveforms_buffers bad mode")

    job_kwargs = fix_job_kwargs(job_kwargs)

    if num_spikes > 0:
        # and run
        func = _worker_distribute_single_buffer
        init_func = _init_worker_distribute_single_buffer

        init_args = (
            recording,
            spikes,
            wf_array_info,
            nbefore,
            nafter,
            return_scaled,
            mode,
            sparsity_mask,
        )
        if job_name is None:
            job_name = f"extract waveforms {mode} mono buffer"

        processor = ChunkRecordingExecutor(
            recording, func, init_func, init_args, job_name=job_name, verbose=verbose, **job_kwargs
        )
        processor.run()

    if mode == "memmap":
        return all_waveforms
    elif mode == "shared_memory":
        if copy:
            if shm is not None:
                # release all sharedmem buffer
                # empty array have None
                shm.unlink()
            return all_waveforms.copy()
        else:
            return all_waveforms, wf_array_info


def _init_worker_distribute_single_buffer(
    recording, spikes, wf_array_info, nbefore, nafter, return_scaled, mode, sparsity_mask
):
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["wf_array_info"] = wf_array_info
    worker_ctx["spikes"] = spikes
    worker_ctx["nbefore"] = nbefore
    worker_ctx["nafter"] = nafter
    worker_ctx["return_scaled"] = return_scaled
    worker_ctx["sparsity_mask"] = sparsity_mask
    worker_ctx["mode"] = mode

    if mode == "memmap":
        filename = wf_array_info["filename"]
        all_waveforms = np.load(str(filename), mmap_mode="r+")
        worker_ctx["all_waveforms"] = all_waveforms
    elif mode == "shared_memory":
        from multiprocessing.shared_memory import SharedMemory

        shm_name, dtype, shape = wf_array_info["shm_name"], wf_array_info["dtype"], wf_array_info["shape"]
        shm = SharedMemory(shm_name)
        all_waveforms = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
        worker_ctx["shm"] = shm
        worker_ctx["all_waveforms"] = all_waveforms

    # prepare segment slices
    segment_slices = []
    for segment_index in range(recording.get_num_segments()):
        s0, s1 = np.searchsorted(spikes["segment_index"], [segment_index, segment_index + 1])
        segment_slices.append((s0, s1))
    worker_ctx["segment_slices"] = segment_slices

    return worker_ctx


# used by ChunkRecordingExecutor
def _worker_distribute_single_buffer(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx["recording"]
    segment_slices = worker_ctx["segment_slices"]
    spikes = worker_ctx["spikes"]
    nbefore = worker_ctx["nbefore"]
    nafter = worker_ctx["nafter"]
    return_scaled = worker_ctx["return_scaled"]
    sparsity_mask = worker_ctx["sparsity_mask"]
    all_waveforms = worker_ctx["all_waveforms"]

    seg_size = recording.get_num_samples(segment_index=segment_index)

    s0, s1 = segment_slices[segment_index]
    in_seg_spikes = spikes[s0:s1]

    # take only spikes in range [start_frame, end_frame]
    # this is a slice so no copy!!
    # the border of segment are protected by nbefore on left an nafter on the right
    i0, i1 = np.searchsorted(
        in_seg_spikes["sample_index"], [max(start_frame, nbefore), min(end_frame, seg_size - nafter)]
    )

    # slice in absolut in spikes vector
    l0 = i0 + s0
    l1 = i1 + s0

    if l1 > l0:
        start = spikes[l0]["sample_index"] - nbefore
        end = spikes[l1 - 1]["sample_index"] + nafter

        # load trace in memory
        traces = recording.get_traces(
            start_frame=start, end_frame=end, segment_index=segment_index, return_scaled=return_scaled
        )

        for spike_index in range(l0, l1):
            sample_index = spikes[spike_index]["sample_index"]
            unit_index = spikes[spike_index]["unit_index"]
            wf = traces[sample_index - start - nbefore : sample_index - start + nafter, :]

            if sparsity_mask is None:
                all_waveforms[spike_index, :, :] = wf
            else:
                mask = sparsity_mask[unit_index, :]
                wf = wf[:, mask]
                all_waveforms[spike_index, :, : wf.shape[1]] = wf

        if worker_ctx["mode"] == "memmap":
            all_waveforms.flush()


def split_waveforms_by_units(unit_ids, spikes, all_waveforms, sparsity_mask=None, folder=None):
    """
    Split a single buffer waveforms into waveforms by units (multi buffers or multi files).

    Parameters
    ----------
    unit_ids: list or numpy array
        List of unit ids
    spikes: numpy array
        The spike vector
    all_waveforms : numpy array
        Single buffer containing all waveforms
    sparsity_mask : None or numpy array
        Optionally the boolean sparsity mask
    folder : None or str or Path
        If a folder is given all waveforms by units are copied in a npy file using f"waveforms_{unit_id}.npy" naming.

    Returns
    -------
    waveforms_by_units: dict of array
        A dict of arrays.
        In case of folder not None, this contain the memmap of the files.
    """
    if folder is not None:
        folder = Path(folder)
    waveforms_by_units = {}
    for unit_index, unit_id in enumerate(unit_ids):
        mask = spikes["unit_index"] == unit_index
        if sparsity_mask is not None:
            chan_mask = sparsity_mask[unit_index, :]
            num_chans = np.sum(chan_mask)
            wfs = all_waveforms[mask, :, :][:, :, :num_chans]
        else:
            wfs = all_waveforms[mask, :, :]

        if folder is None:
            waveforms_by_units[unit_id] = wfs
        else:
            np.save(folder / f"waveforms_{unit_id}.npy", wfs)
            # this avoid keeping in memory all waveforms
            waveforms_by_units[unit_id] = np.load(f"waveforms_{unit_id}.npy", mmap_mode="r")

    return waveforms_by_units


def has_exceeding_spikes(sorting, recording) -> bool:
    """
    Check if the sorting objects has spikes exceeding the recording number of samples, for all segments

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object
    recording : BaseRecording
        The recording object

    Returns
    -------
    bool
        True if exceeding spikes, False otherwise
    """
    spike_vector = sorting.to_spike_vector()
    for segment_index in range(recording.get_num_segments()):
        start_seg_ind, end_seg_ind = np.searchsorted(spike_vector["segment_index"], [segment_index, segment_index + 1])
        spike_vector_seg = spike_vector[start_seg_ind:end_seg_ind]
        if len(spike_vector_seg) > 0:
            if spike_vector_seg["sample_index"][-1] > recording.get_num_samples(segment_index=segment_index) - 1:
                return True
            if spike_vector_seg["sample_index"][0] < 0:
                return True
    return False


def estimate_templates(
    recording: BaseRecording,
    spikes: np.ndarray,
    unit_ids: list | np.ndarray,
    nbefore: int,
    nafter: int,
    operator: str = "average",
    return_scaled: bool = True,
    job_name=None,
    **job_kwargs,
):
    """
    Estimate dense templates with "average" or "median".
    If "average" internally estimate_templates_with_accumulator() is used to saved memory/

    Parameters
    ----------

    recording: BaseRecording
        The recording object
    spikes: 1d numpy array with several fields
        Spikes handled as a unique vector.
        This vector can be obtained with: `spikes = sorting.to_spike_vector()`
    unit_ids: list ot numpy
        List of unit_ids
    nbefore: int
        Number of samples to cut out before a spike
    nafter: int
        Number of samples to cut out after a spike
    return_scaled: bool, default: True
        If True, the traces are scaled before averaging

    Returns
    -------
    templates_array: np.array
        The average templates with shape (num_units, nbefore + nafter, num_channels)

    """

    if job_name is None:
        job_name = "estimate_templates"

    if operator == "average":
        templates_array = estimate_templates_with_accumulator(
            recording, spikes, unit_ids, nbefore, nafter, return_scaled=return_scaled, job_name=job_name, **job_kwargs
        )
    elif operator == "median":
        all_waveforms, wf_array_info = extract_waveforms_to_single_buffer(
            recording,
            spikes,
            unit_ids,
            nbefore,
            nafter,
            mode="shared_memory",
            return_scaled=return_scaled,
            copy=False,
            **job_kwargs,
        )
        templates_array = np.zeros(
            (len(unit_ids), all_waveforms.shape[1], all_waveforms.shape[2]), dtype=all_waveforms.dtype
        )
        for unit_index, unit_id in enumerate(unit_ids):
            wfs = all_waveforms[spikes["unit_index"] == unit_index]
            templates_array[unit_index, :, :] = np.median(wfs, axis=0)
        # release shared memory after the median
        wf_array_info["shm"].unlink()

    else:
        raise ValueError(f"estimate_templates(..., operator={operator}) wrong operator must be average or median")

    return templates_array


def estimate_templates_with_accumulator(
    recording: BaseRecording,
    spikes: np.ndarray,
    unit_ids: list | np.ndarray,
    nbefore: int,
    nafter: int,
    return_scaled: bool = True,
    job_name=None,
    return_std: bool = False,
    verbose: bool = False,
    **job_kwargs,
):
    """
    This is a fast implementation to compute template averages and standard deviations.
    This is useful to estimate sparsity without the need to allocate large waveform buffers.
    The mechanism is pretty simple: it accumulates and sums spike waveforms (and their squared)
    in-place per worker and per unit.
    Note that median and percentiles can't be computed with this method, because they don't support
    the accumulator implementation.

    Parameters
    ----------
    recording: BaseRecording
        The recording object
    spikes: 1d numpy array with several fields
        Spikes handled as a unique vector.
        This vector can be obtained with: `spikes = sorting.to_spike_vector()`
    unit_ids: list ot numpy
        List of unit_ids
    nbefore: int
        Number of samples to cut out before a spike
    nafter: int
        Number of samples to cut out after a spike
    return_scaled: bool, default: True
        If True, the traces are scaled before averaging
    return_std: bool, default: False
        If True, the standard deviation is also computed.

    Returns
    -------
    templates_array: np.array
        The average templates with shape (num_units, nbefore + nafter, num_channels)
    """

    assert spikes.size > 0, "estimate_templates() need non empty sorting"

    job_kwargs = fix_job_kwargs(job_kwargs)
    num_worker = job_kwargs["n_jobs"]

    num_chans = recording.get_num_channels()
    num_units = len(unit_ids)

    shape = (num_worker, num_units, nbefore + nafter, num_chans)
    dtype = np.dtype("float32")
    waveform_accumulator_per_worker, shm = make_shared_array(shape, dtype)
    shm_name = shm.name
    if return_std:
        waveform_squared_accumulator_per_worker, shm_squared = make_shared_array(shape, dtype)
        shm_squared_name = shm_squared.name
    else:
        waveform_squared_accumulator_per_worker = None
        shm_squared_name = None

    # trick to get the work_index given pid arrays
    lock = multiprocessing.Lock()
    array_pid = multiprocessing.Array("i", num_worker)
    for i in range(num_worker):
        array_pid[i] = -1

    func = _worker_estimate_templates
    init_func = _init_worker_estimate_templates

    init_args = (
        recording,
        spikes,
        shm_name,
        shm_squared_name,
        shape,
        dtype,
        nbefore,
        nafter,
        return_scaled,
        lock,
        array_pid,
    )

    if job_name is None:
        job_name = "estimate_templates_with_accumulator"
    processor = ChunkRecordingExecutor(
        recording, func, init_func, init_args, job_name=job_name, verbose=verbose, **job_kwargs
    )
    processor.run()

    # average
    waveforms_sum = np.sum(waveform_accumulator_per_worker, axis=0)
    if return_std:
        # we need a copy here because we will use the means to compute the stds
        template_means = waveforms_sum.copy()
    else:
        # waveforms_sum will also be changed in this case when acting on template_means
        template_means = waveforms_sum

    unit_indices, spike_count = np.unique(spikes["unit_index"], return_counts=True)
    template_means[unit_indices, :, :] /= spike_count[:, np.newaxis, np.newaxis]

    if return_std:
        waveforms_squared_sum = np.sum(waveform_squared_accumulator_per_worker, axis=0)
        # standard deviation
        template_stds = np.zeros_like(template_means)
        for unit_index, count in zip(unit_indices, spike_count):
            residuals = (
                waveforms_squared_sum[unit_index] - 2 * template_means[unit_index] * waveforms_sum[unit_index]
            ) + count * template_means[unit_index] ** 2
            residuals[residuals < 0] = 0
            template_stds[unit_index] = np.sqrt(residuals / count)
        del waveform_squared_accumulator_per_worker
        shm_squared.unlink()
        shm_squared.close()

    # important : release the sharedmem
    del waveform_accumulator_per_worker
    shm.unlink()
    shm.close()

    if return_std:
        return template_means, template_stds
    else:
        return template_means


def _init_worker_estimate_templates(
    recording,
    spikes,
    shm_name,
    shm_squared_name,
    shape,
    dtype,
    nbefore,
    nafter,
    return_scaled,
    lock,
    array_pid,
):
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["spikes"] = spikes
    worker_ctx["nbefore"] = nbefore
    worker_ctx["nafter"] = nafter
    worker_ctx["return_scaled"] = return_scaled

    from multiprocessing.shared_memory import SharedMemory
    import multiprocessing

    shm = SharedMemory(shm_name)
    waveform_accumulator_per_worker = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)

    worker_ctx["shm"] = shm
    worker_ctx["waveform_accumulator_per_worker"] = waveform_accumulator_per_worker
    if shm_squared_name is not None:
        shm_squared = SharedMemory(shm_squared_name)
        waveform_squared_accumulator_per_worker = np.ndarray(shape=shape, dtype=dtype, buffer=shm_squared.buf)
        worker_ctx["shm_squared"] = shm_squared
        worker_ctx["waveform_squared_accumulator_per_worker"] = waveform_squared_accumulator_per_worker

    # prepare segment slices
    segment_slices = []
    for segment_index in range(recording.get_num_segments()):
        s0, s1 = np.searchsorted(spikes["segment_index"], [segment_index, segment_index + 1])
        segment_slices.append((s0, s1))
    worker_ctx["segment_slices"] = segment_slices

    child_process = multiprocessing.current_process()

    lock.acquire()
    num_worker = None
    for i in range(len(array_pid)):
        if array_pid[i] == -1:
            num_worker = i
            array_pid[i] = child_process.ident
            break
    worker_ctx["worker_index"] = num_worker
    lock.release()

    return worker_ctx


# used by ChunkRecordingExecutor
def _worker_estimate_templates(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx["recording"]
    segment_slices = worker_ctx["segment_slices"]
    spikes = worker_ctx["spikes"]
    nbefore = worker_ctx["nbefore"]
    nafter = worker_ctx["nafter"]
    waveform_accumulator_per_worker = worker_ctx["waveform_accumulator_per_worker"]
    waveform_squared_accumulator_per_worker = worker_ctx.get("waveform_squared_accumulator_per_worker", None)
    worker_index = worker_ctx["worker_index"]
    return_scaled = worker_ctx["return_scaled"]

    seg_size = recording.get_num_samples(segment_index=segment_index)

    s0, s1 = segment_slices[segment_index]
    in_seg_spikes = spikes[s0:s1]

    # take only spikes in range [start_frame, end_frame]
    # this is a slice so no copy!!
    # the border of segment are protected by nbefore on left an nafter on the right
    i0, i1 = np.searchsorted(
        in_seg_spikes["sample_index"], [max(start_frame, nbefore), min(end_frame, seg_size - nafter)]
    )

    # slice in absolut in spikes vector
    l0 = i0 + s0
    l1 = i1 + s0

    if l1 > l0:
        start = spikes[l0]["sample_index"] - nbefore
        end = spikes[l1 - 1]["sample_index"] + nafter

        # load trace in memory
        traces = recording.get_traces(
            start_frame=start, end_frame=end, segment_index=segment_index, return_scaled=return_scaled
        )

        for spike_index in range(l0, l1):
            sample_index = spikes[spike_index]["sample_index"]
            unit_index = spikes[spike_index]["unit_index"]
            wf = traces[sample_index - start - nbefore : sample_index - start + nafter, :]

            waveform_accumulator_per_worker[worker_index, unit_index, :, :] += wf
            if waveform_squared_accumulator_per_worker is not None:
                waveform_squared_accumulator_per_worker[worker_index, unit_index, :, :] += wf**2
