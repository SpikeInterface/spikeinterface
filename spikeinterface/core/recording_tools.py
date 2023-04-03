import numpy as np


def get_random_data_chunks(
    recording,
    return_scaled=False,
    num_chunks_per_segment=20,
    chunk_size=10000,
    concatenated=True,
    seed=0,
    margin_frames=0,
):
    """
    Exctract random chunks across segments

    This is used for instance in get_noise_levels() to estimate noise on traces.

    Parameters
    ----------
    recording: BaseRecording
        The recording to get random chunks from
    return_scaled: bool
        If True, returned chunks are scaled to uV
    num_chunks_per_segment: int
        Number of chunks per segment
    chunk_size: int
        Size of a chunk in number of frames
    concatenated: bool (default True)
        If True chunk are concatenated along time axis.
    seed: int
        Random seed
    Returns
    -------
    chunk_list: np.array
        Array of concatenate chunks per segment
    """
    # TODO: if segment have differents length make another sampling that dependant on the length of the segment
    # Should be done by changing kwargs with total_num_chunks=XXX and total_duration=YYYY
    # And randomize the number of chunk per segment weighted by segment duration

    # check chunk size
    for segment_index in range(recording.get_num_segments()):
        assert chunk_size < recording.get_num_samples(segment_index), (
            f"chunk_size is greater than the number "
            f"of samples for segment index {segment_index}. "
            f"Use a smaller chunk_size!"
        )

    chunk_list = []
    for segment_index in range(recording.get_num_segments()):
        length = recording.get_num_frames(segment_index)

        random_starts = np.random.RandomState(seed=seed).randint(
            margin_frames,
            length - chunk_size - margin_frames,
            size=num_chunks_per_segment,
        )
        for start_frame in random_starts:
            chunk = recording.get_traces(
                start_frame=start_frame,
                end_frame=start_frame + chunk_size,
                segment_index=segment_index,
                return_scaled=return_scaled,
            )
            chunk_list.append(chunk)
    if concatenated:
        return np.concatenate(chunk_list, axis=0)
    else:
        return chunk_list


def get_channel_distances(recording):
    """
    Distance between channel pairs
    """
    locations = recording.get_channel_locations()
    channel_distances = np.linalg.norm(
        locations[:, np.newaxis] - locations[np.newaxis, :], axis=2
    )

    return channel_distances


def get_closest_channels(recording, channel_ids=None, num_channels=None):
    """Get closest channels + distances

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to get closest channels
    channel_ids: list
        List of channels ids to compute there near neighborhood
    num_channels: int, optional
        Maximum number of neighborhood channels to return

    Returns
    -------
    closest_channels_inds : array (2d)
        Closest channel indices in ascending order for each channel id given in input
    dists: array (2d)
        Distance in ascending order for each channel id given in input
    """
    if channel_ids is None:
        channel_ids = recording.get_channel_ids()
    if num_channels is None:
        num_channels = len(channel_ids) - 1

    locations = recording.get_channel_locations(channel_ids=channel_ids)

    closest_channels_inds = []
    dists = []
    for i in range(locations.shape[0]):
        distances = np.linalg.norm(locations[i, :] - locations, axis=1)
        order = np.argsort(distances)
        closest_channels_inds.append(order[1 : num_channels + 1])
        dists.append(distances[order][1 : num_channels + 1])

    return np.array(closest_channels_inds), np.array(dists)


def get_noise_levels(recording, return_scaled=True, **random_chunk_kwargs):
    """
    Estimate noise for each channel using MAD methods.

    Internally it sample some chunk across segment.
    And then, it use MAD estimator (more robust than STD)

    """
    random_chunks = get_random_data_chunks(
        recording, return_scaled=return_scaled, **random_chunk_kwargs
    )
    med = np.median(random_chunks, axis=0, keepdims=True)
    # hard-coded so that core doesn't depend on scipy
    noise_levels = (
        np.median(np.abs(random_chunks - med), axis=0) / 0.6744897501960817
    )
    return noise_levels


def get_chunk_with_margin(
    rec_segment,
    start_frame,
    end_frame,
    channel_indices,
    margin,
    add_zeros=False,
    add_reflect_padding=False,
    window_on_margin=False,
    dtype=None,
):
    """
    Helper to get chunk with margin

    The margin is extracted from the recording when possible. If
    at the edge of the recording, no margin is used unless one
    of `add_zeros` or `add_reflect_padding` is True. In the first
    case zero padding is used, in the second case np.pad is called
    with mod="reflect".
    """
    length = rec_segment.get_num_samples()

    if channel_indices is None:
        channel_indices = slice(None)

    if not (add_zeros or add_reflect_padding):
        if window_on_margin and not add_zeros:
            raise ValueError("window_on_margin requires add_zeros=True")
        if start_frame is None:
            left_margin = 0
            start_frame = 0
        elif start_frame < margin:
            left_margin = start_frame
        else:
            left_margin = margin

        if end_frame is None:
            right_margin = 0
            end_frame = length
        elif end_frame > (length - margin):
            right_margin = length - end_frame
        else:
            right_margin = margin
        traces_chunk = rec_segment.get_traces(
            start_frame - left_margin,
            end_frame + right_margin,
            channel_indices,
        )

    else:
        # either add_zeros or reflect_padding
        assert start_frame is not None
        assert end_frame is not None
        chunk_size = end_frame - start_frame
        full_size = chunk_size + 2 * margin

        if start_frame < margin:
            start_frame2 = 0
            left_pad = margin - start_frame
        else:
            start_frame2 = start_frame - margin
            left_pad = 0

        if end_frame > (length - margin):
            end_frame2 = length
            right_pad = end_frame + margin - length
        else:
            end_frame2 = end_frame + margin
            right_pad = 0

        traces_chunk = rec_segment.get_traces(
            start_frame2, end_frame2, channel_indices
        )

        if (
            dtype is not None
            or window_on_margin
            or left_pad > 0
            or right_pad > 0
        ):
            need_copy = True
        else:
            need_copy = False

        left_margin = margin
        right_margin = margin

        if need_copy:
            if dtype is None:
                dtype = traces_chunk.dtype

            left_margin = margin
            if end_frame < (length + margin):
                right_margin = margin
            else:
                right_margin = end_frame + margin - length

            if add_zeros:
                traces_chunk2 = np.zeros(
                    (full_size, traces_chunk.shape[1]), dtype=dtype
                )
                i0 = left_pad
                i1 = left_pad + traces_chunk.shape[0]
                traces_chunk2[i0:i1, :] = traces_chunk
                if window_on_margin:
                    # apply inplace taper on border
                    taper = (
                        1 - np.cos(np.arange(margin) / margin * np.pi)
                    ) / 2
                    taper = taper[:, np.newaxis]
                    traces_chunk2[:margin] *= taper
                    traces_chunk2[-margin:] *= taper[::-1]
                traces_chunk = traces_chunk2
            elif add_reflect_padding:
                # in this case, we don't want to taper
                traces_chunk = np.pad(
                    traces_chunk.astype(dtype),
                    [(left_pad, right_pad), (0, 0)],
                    mode="reflect",
                )
            else:
                # we need a copy to change the dtype
                traces_chunk = np.asarray(traces_chunk, dtype=dtype)

    return traces_chunk, left_margin, right_margin


def order_channels_by_depth(
    recording, channel_ids=None, dimensions=("x", "y")
):
    """
    Order channels by depth, by first ordering the x-axis, and then the y-axis.

    Parameters
    ----------
    recording : BaseRecording
        The input recording
    channel_ids : list/array or None
        If given, a subset of channels to order locations for
    dimensions : str or tuple
        If str, it needs to be 'x', 'y', 'z'.
        If tuple, it sorts the locations in two dimensions using lexsort.
        This approach is recommended since there is less ambiguity, by default ('x', 'y')

    Returns
    -------
    order_f : np.array
        Array with sorted indices
    order_r : np.array
        Array with indices to revert sorting
    """
    locations = recording.get_channel_locations()
    ndim = locations.shape[1]
    channel_inds = recording.ids_to_indices(ids=channel_ids, prefer_slice=True)
    locations = locations[channel_inds, :]

    if isinstance(dimensions, str):
        dim = ["x", "y", "z"].index(dimensions)
        assert dim < ndim, "Invalid dimensions!"
        order_f = np.argsort(locations[:, dim], kind="stable")
    else:
        assert isinstance(
            dimensions, tuple
        ), "dimensions can be a str or a tuple"
        locations_to_sort = ()
        for dim in dimensions:
            dim = ["x", "y", "z"].index(dim)
            assert dim < ndim, "Invalid dimensions!"
            locations_to_sort += (locations[:, dim],)
        order_f = np.lexsort(locations_to_sort)
    order_r = np.argsort(order_f, kind="stable")

    return order_f, order_r


def check_probe_do_not_overlap(probes):
    """
    When several probes this check that that they do not overlap in space
    and so channel positions can be safly concatenated.
    """
    for i in range(len(probes)):
        probe_i = probes[i]
        # check that all positions in probe_j are outside probe_i boundaries
        x_bounds_i = [
            np.min(probe_i.contact_positions[:, 0]),
            np.max(probe_i.contact_positions[:, 0]),
        ]
        y_bounds_i = [
            np.min(probe_i.contact_positions[:, 1]),
            np.max(probe_i.contact_positions[:, 1]),
        ]

        for j in range(i + 1, len(probes)):
            probe_j = probes[j]

            if np.any(
                np.array(
                    [
                        x_bounds_i[0] < cp[0] < x_bounds_i[1]
                        and y_bounds_i[0] < cp[1] < y_bounds_i[1]
                        for cp in probe_j.contact_positions
                    ]
                )
            ):
                raise Exception(
                    "Probes are overlapping! Retrieve locations of single probes separately"
                )
