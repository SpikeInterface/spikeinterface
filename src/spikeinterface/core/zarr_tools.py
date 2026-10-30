import warnings
import json
import pickle
import numpy as np
import zarr

from spikeinterface.core.job_tools import ensure_chunk_size

# metadata keys that are not members of a group (zarr v2 and v3)
_ZARR_METADATA_KEYS = (".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json")


def adjust_chunks_shards_and_job_kwargs(
    chunks=None, extra_chunks=None, shards=None, shard_factor=None, job_kwargs=None, time_series=None
):
    """
    Adjust chunks, shards, and job_kwargs for zarr storage.

    Parameters
    ----------
    chunks : tuple or None
        Chunking dimensions for the zarr dataset.
    extra_chunks : tuple or None
        Extra chunking dimensions for the zarr dataset.
    shards : tuple or None
        Sharding configuration for the zarr dataset.
    shard_factor : int or tuple or None
        Factor to determine shard sizes based on chunks.
    job_kwargs : dict or None
        Job-related keyword arguments, including 'chunk_size'.
    time_series : TimeSeries or None
        The time series object being stored.

    Returns
    -------
    chunks : tuple
        Adjusted chunking dimensions.
    shards : tuple or None
        Adjusted sharding configuration.
    job_kwargs : dict
        Updated job-related keyword arguments.
    """

    # Chunking and sharding
    if shards is not None and shard_factor is not None:
        raise ValueError("Cannot specify both 'shards' and 'shard_factor' in zarr_kwargs")
    if chunks is not None and extra_chunks is not None:
        raise ValueError("Cannot specify both 'chunks' and 'extra_chunks' in zarr_kwargs")

    # If not specified by chunk, we set the chunk size in the first dimension (time) to be the chunk size that we use
    # for the job executor, and the chunk size in the second dimension (channels) to be either the provided
    # channel_chunk_size or the total number of channels (no chunking in channels).
    if chunks is not None:
        job_kwargs["chunk_size"] = chunks[0]
    else:
        chunk_size = ensure_chunk_size(time_series, **job_kwargs)
        chunks = (
            (chunk_size,) + extra_chunks if extra_chunks is not None else (chunk_size,) + time_series.get_shape(0)[1:]
        )

    if shards is not None:
        assert len(shards) == len(chunks), "Shards and chunks must have the same number of dimensions"
        for dim in range(len(chunks)):
            assert (
                shards[dim] >= chunks[dim] and shards[dim] % chunks[dim] == 0
            ), "Shard size must be a multiple of chunk size"
        # When sharding is used, we set the `num_chunk_per_batch` to match the number of chunks in
        # the shard, so that each job writes one shard
        job_kwargs["num_chunk_per_batch"] = int(shards[0] // chunks[0])
    elif shard_factor is not None:
        # If shard_factor is an integer, we only apply it to the first dimension. If it's an iterable,
        # we apply it to all dimensions.
        if isinstance(shard_factor, (int, np.integer)):
            shards = (chunks[0] * shard_factor,) + chunks[1:]
        else:
            if len(shard_factor) != len(chunks):
                raise ValueError("shard_factor must have the same length as chunks when it is an iterable")
            shards = tuple(chunks[dim] * shard_factor[dim] for dim in range(len(chunks)))
        job_kwargs["num_chunk_per_batch"] = int(shards[0] // chunks[0])

    return chunks, shards, job_kwargs


def check_compressors_match(comp1, comp2, skip_typesize=True):
    """
    Check if two compressor objects match.

    Parameters
    ----------
    comp1 : zarr.abc.codec.Codec | Tuple[zarr.abc.codec.Codec]
        The first compressor object to compare.
    comp2 : zarr.abc.codec.Codec | Tuple[zarr.abc.codec.Codec]
        The second compressor object to compare.
    skip_typesize : bool, optional
        Whether to skip the typesize check, default: True
    """
    if not isinstance(comp1, (list, tuple)):
        assert not isinstance(comp2, list)
        comp1 = [comp1]
        comp2 = [comp2]
    for i in range(len(comp1)):
        comp1_dict = comp1[i].to_dict()
        comp2_dict = comp2[i].to_dict()
        if skip_typesize:
            if "typesize" in comp1_dict["configuration"]:
                comp1_dict["configuration"].pop("typesize", None)
        if "typesize" in comp2_dict["configuration"]:
            comp2_dict["configuration"].pop("typesize", None)
        assert comp1_dict == comp2_dict, f"Compressor {i} does not match: {comp1_dict} != {comp2_dict}"


class LegacyZarrObjectArray:
    """
    Read-only stand-in for a zarr v2 object-dtype array that zarr-python >= 3 cannot open.

    spikeinterface < 0.105 saved python objects (dicts, lists, provenance dicts, ...) as
    length-1 object-dtype arrays using the `numcodecs` JSON or Pickle object codecs.
    zarr-python >= 3 dropped support for these dtype/codec combinations, so such arrays
    are decoded "manually" with `numcodecs` and exposed through this minimal wrapper,
    which mimics the small part of the `zarr.Array` API used to read them
    (`attrs`, `__getitem__`, `shape`, `dtype`, `len`).

    Parameters
    ----------
    values : np.ndarray
        The decoded object-dtype array.
    attrs : dict
        The zarr attributes of the array (content of `.zattrs`).
    """

    def __init__(self, values: np.ndarray, attrs: dict):
        self._values = values
        self._attrs = dict(attrs)

    @property
    def attrs(self) -> dict:
        return self._attrs

    @property
    def shape(self):
        return self._values.shape

    @property
    def dtype(self):
        return self._values.dtype

    def __len__(self):
        return len(self._values)

    def __getitem__(self, key):
        return self._values[key]

    def __array__(self, dtype=None, copy=None):
        if dtype is None:
            return self._values
        return self._values.astype(dtype)

    def __repr__(self):
        return f"LegacyZarrObjectArray(shape={self.shape}, attrs={list(self._attrs.keys())})"


def _sync(coroutine):
    """Run a zarr async coroutine synchronously (zarr-python >= 3)."""
    from zarr.core.sync import sync

    return sync(coroutine)


def _store_get_json(store, key: str):
    """Read and json-decode a metadata key from a zarr store. Return None if missing."""
    import json

    from zarr.core.buffer import default_buffer_prototype

    buffer = _sync(store.get(key, prototype=default_buffer_prototype()))
    if buffer is None:
        return None
    return json.loads(buffer.to_bytes().decode())


def _store_get_bytes(store, key: str):
    """Read raw bytes from a zarr store. Return None if the key is missing."""
    from zarr.core.buffer import default_buffer_prototype

    buffer = _sync(store.get(key, prototype=default_buffer_prototype()))
    if buffer is None:
        return None
    return buffer.to_bytes()


def read_legacy_zarr_object_array(store, path: str) -> LegacyZarrObjectArray:
    """
    Read a zarr v2 object-dtype array written with a `numcodecs` object codec
    (JSON, Pickle, MsgPack), which zarr-python >= 3 cannot open.

    Only single-chunk arrays are supported, which is what spikeinterface < 0.105 wrote.

    Parameters
    ----------
    store : zarr.abc.store.Store
        The store containing the array.
    path : str
        The path of the array inside the store.

    Returns
    -------
    legacy_array : LegacyZarrObjectArray
        The decoded array.
    """
    import numcodecs

    path = path.strip("/")
    zarray = _store_get_json(store, f"{path}/.zarray")
    if zarray is None:
        raise KeyError(f"No zarr v2 array metadata found at {path}")
    zattrs = _store_get_json(store, f"{path}/.zattrs") or {}

    shape = tuple(zarray["shape"])
    chunks = tuple(zarray["chunks"])
    if any(chunk_size < dim for chunk_size, dim in zip(chunks, shape)):
        raise NotImplementedError(
            f"Legacy zarr v2 object array at {path} has more than one chunk, which is not supported"
        )

    separator = zarray.get("dimension_separator", ".")
    chunk_key = separator.join(["0"] * max(len(shape), 1))
    chunk_bytes = _store_get_bytes(store, f"{path}/{chunk_key}")
    if chunk_bytes is None:
        # array was never written to: return an empty object array
        return LegacyZarrObjectArray(np.empty(shape, dtype=object), zattrs)

    if zarray.get("compressor", None) is not None:
        chunk_bytes = numcodecs.get_codec(zarray["compressor"]).decode(chunk_bytes)
    values = chunk_bytes
    # filters are applied on encode, so they are decoded in reverse order.
    # for object arrays the object codec is the (only) filter
    for filter_config in reversed(zarray.get("filters", None) or []):
        if filter_config.get("id") == "pickle":
            # numcodecs>=0.16 disables Pickle.decode by default; we already trust pickles in the analyzer folder
            values = pickle.loads(bytes(values))
        else:
            values = numcodecs.get_codec(filter_config).decode(values)

    values = np.asarray(values, dtype=object).reshape(shape)

    return LegacyZarrObjectArray(values, zattrs)


def get_zarr_attr_or_legacy_object(zarr_group, name: str):
    """
    Get a value saved either in the attributes of a zarr group (spikeinterface >= 0.105)
    or as a legacy zarr v2 length-1 object array with the same name
    (spikeinterface < 0.105, e.g. "recording" and "sorting_provenance").

    Parameters
    ----------
    zarr_group : zarr.Group
        The zarr group to read from.
    name : str
        The name of the attribute / legacy array.

    Returns
    -------
    value : Any | None
        The value, or None if it is not found (or cannot be decoded).
    """
    value = zarr_group.attrs.get(name, None)
    if value is not None:
        return value
    try:
        legacy_array = read_legacy_zarr_object_array(zarr_group.store, f"{zarr_group.path}/{name}")
    except Exception:
        return None
    if len(legacy_array) == 0:
        return None
    return legacy_array[0]


def _list_member_names(zarr_group) -> list[str]:
    """List the member names of a zarr group without opening them."""
    consolidated = getattr(zarr_group.metadata, "consolidated_metadata", None)
    if consolidated is not None:
        return list(consolidated.metadata.keys())

    store = zarr_group.store
    if not store.supports_listing:
        raise ValueError(
            f"The store associated to this group ({type(store).__name__}) does not support listing, "
            "so its members cannot be listed without consolidated metadata."
        )

    async def _list_dir():
        return [key async for key in store.list_dir(zarr_group.path)]

    keys = _sync(_list_dir())
    # skip zarr metadata documents and hidden files (e.g. AppleDouble "._*" files)
    return sorted(key for key in keys if key not in _ZARR_METADATA_KEYS and not key.startswith("."))


def iterate_zarr_group(zarr_group, skip_unreadable: bool = True):
    """
    Iterate over the members (arrays and sub-groups) of a zarr group, transparently
    handling zarr v2 and zarr v3 groups.

    Unlike `zarr.Group.keys()` / `zarr.Group.members()`, a single member that
    zarr-python >= 3 cannot open does not make the whole iteration fail. This happens for
    zarr v2 data saved by spikeinterface < 0.105, where python objects were stored as
    object-dtype arrays with the `numcodecs` JSON/Pickle object codecs. Such arrays are
    decoded with `numcodecs` and returned as a `LegacyZarrObjectArray`.

    Parameters
    ----------
    zarr_group : zarr.Group
        The zarr group to iterate over.
    skip_unreadable : bool, default: True
        If True, members that cannot be opened nor decoded are skipped with a warning.
        If False, an error is raised instead.

    Yields
    ------
    name : str
        The name of the member.
    member : zarr.Group | zarr.Array | LegacyZarrObjectArray
        The member itself.
    """
    for name in _list_member_names(zarr_group):
        # note: members are retrieved outside of the yield statement, so that exceptions
        # raised by the consumer of this generator are not caught here
        member = None
        open_error = None
        try:
            member = zarr_group[name]
        except KeyError:
            # the key is an object in the store (e.g. a stray file), not a zarr node
            continue
        except Exception as e:
            open_error = e

        if open_error is not None:
            # the member exists but zarr-python cannot open it: try the legacy object array path
            try:
                member = read_legacy_zarr_object_array(zarr_group.store, f"{zarr_group.path}/{name}")
            except Exception as legacy_error:
                if not skip_unreadable:
                    raise ValueError(
                        f"Cannot read member '{name}' of zarr group '{zarr_group.path}': "
                        f"{open_error}\nLegacy object array fallback failed with: {legacy_error}"
                    ) from open_error
                warnings.warn(
                    f"Skipping member '{name}' of zarr group '{zarr_group.path}' because it cannot be read "
                    f"with zarr v{zarr.__version__}: {open_error} ({legacy_error})"
                )
                continue

        yield name, member


def get_zarr_group_keys(zarr_group) -> list[str]:
    """
    Return the names of the members of a zarr group, for both zarr v2 and zarr v3 groups.

    Contrary to `zarr.Group.keys()`, the members are not opened, so this also works for
    groups containing legacy zarr v2 arrays that zarr-python >= 3 cannot open
    (see `iterate_zarr_group`).

    Parameters
    ----------
    zarr_group : zarr.Group
        The zarr group to list.

    Returns
    -------
    keys : list[str]
        The names of the members of the group.
    """
    return _list_member_names(zarr_group)


def write_json_array(group, name: str, data, compressors=None, filters=None):
    """
    Save json-serializable data as the utf-8 bytes of its json text in a uint8 array.

    This replaces the length-1 object arrays with a json codec, which zarr-python 3 cannot write.
    The data is stored as array data rather than as an attribute, so it is only read when needed:
    attributes are read every time the group is opened, which is slow for large data such as a
    recording dict. The array is marked with an "encoding" attribute, so that other encodings can be
    added later. See `read_encoded_array` for the reverse operation.

    Parameters
    ----------
    group : zarr.Group
        The zarr group in which the array is created.
    name : str
        The name of the array.
    data : dict | list
        The json-serializable data to save.
    compressors: BytesBytesCodec | ArrayBytesCodec | list of them, or None, default: None
        Compressors to use for the array.
    filters: ArrayArrayCodec | list[ArrayArrayCodec] or None, default: None
        Filters to use for the array.

    Returns
    -------
    array : zarr.Array
        The created array.
    """
    payload = np.frombuffer(json.dumps(data).encode("utf-8"), dtype="uint8")
    codec_kwargs = build_codec_pipeline(filters=filters, compressors=compressors)
    array = group.create_array(name=name, data=payload, **codec_kwargs)
    array.attrs["encoding"] = "json"
    return array


def write_pickle_array(group, name: str, data, compressors=None, filters=None):
    """
    Save a python object as its pickle bytes in a uint8 array.

    This replaces the length-1 object arrays with a pickle codec, which zarr-python 3 cannot write.
    It is used for extension data that is not json-serializable, such as the PCA models of the
    "principal_components" extension. See `read_encoded_array` for the reverse operation.

    Parameters
    ----------
    group : zarr.Group
        The zarr group in which the array is created.
    name : str
        The name of the array.
    data : Any
        The picklable object to save.
    compressors: BytesBytesCodec | ArrayBytesCodec | list of them, or None, default: None
        Compressors to use for the array.
    filters: ArrayArrayCodec | list[ArrayArrayCodec] or None, default: None
        Filters to use for the array.

    Returns
    -------
    array : zarr.Array
        The created array.
    """
    payload = np.frombuffer(pickle.dumps(data), dtype="uint8")
    codec_kwargs = build_codec_pipeline(filters=filters, compressors=compressors)
    array = group.create_array(name=name, data=payload, **codec_kwargs)
    array.attrs["encoding"] = "pickle"
    return array


def read_encoded_array(array):
    """
    Read data saved by `write_json_array` or `write_pickle_array`, or a legacy length-1 object array
    (json or pickle codec) saved by spikeinterface<=0.105.

    Parameters
    ----------
    array : zarr.Array
        The array to read.

    Returns
    -------
    data : Any
        The saved data.
    """
    encoding = array.attrs.get("encoding", None)
    if encoding is None:
        # legacy length-1 object array
        return array[0]
    if encoding == "json":
        return json.loads(np.asarray(array[:]).tobytes().decode("utf-8"))
    if encoding == "pickle":
        return pickle.loads(np.asarray(array[:]).tobytes())
    raise ValueError(
        f"Unknown encoding {encoding!r} for {array.name}. It was probably saved by a newer version of spikeinterface."
    )


def build_codec_pipeline(filters=None, compressors=None):
    """
    Build zarr v3 codec kwargs from filters and compressors.

    Classifies codecs into the three slots accepted by ``zarr.Group.create_array()``:
      1. ``filters``    — ArrayArrayCodec  (e.g. Delta)
      2. ``serializer`` — ArrayBytesCodec  (e.g. WavPack, BytesCodec)
      3. ``compressors``— BytesBytesCodec  (e.g. BloscCodec, ZstdCodec)

    This allows callers to pass an ArrayBytesCodec (e.g. WavPack) as a
    compressor and have it placed in the correct serializer slot automatically.

    Parameters
    ----------
    filters : ArrayArrayCodec or list of ArrayArrayCodec or None
        Codec(s) applied before serialization.
    compressors : codec or list of codecs or None
        Can be a mix of ArrayBytesCodec (serializer) and BytesBytesCodec
        (byte-level compressors). At most one ArrayBytesCodec is allowed.

    Returns
    -------
    dict
        Keyword arguments to unpack into ``zarr.Group.create_array()``.
        Only keys with explicit values are included; omitted keys let zarr
        use its defaults.

    Raises
    ------
    ValueError
        If filters contain non-ArrayArrayCodec instances, if more than one
        ArrayBytesCodec is provided, or if an unrecognised codec type is passed.
    """
    from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec

    if filters is None:
        filters = []
    if not isinstance(filters, (list, tuple)):
        filters = [filters]

    if compressors is None:
        compressors = []
    if not isinstance(compressors, (list, tuple)):
        compressors = [compressors]

    for f in filters:
        if not isinstance(f, ArrayArrayCodec):
            raise ValueError(f"All filters must be ArrayArrayCodec instances, got {type(f)}")

    serializers = [c for c in compressors if isinstance(c, ArrayBytesCodec)]
    byte_compressors = [c for c in compressors if isinstance(c, BytesBytesCodec)]
    invalid = [c for c in compressors if not isinstance(c, (ArrayBytesCodec, BytesBytesCodec))]

    if invalid:
        raise ValueError(
            f"Compressors must be ArrayBytesCodec or BytesBytesCodec instances, got {[type(c) for c in invalid]}"
        )
    if len(serializers) > 1:
        raise ValueError("Only one ArrayBytesCodec (serializer) is allowed in the codec pipeline.")

    codec_kwargs = {}
    codec_kwargs["filters"] = filters
    codec_kwargs["serializer"] = serializers[0] if len(serializers) == 1 else "auto"
    codec_kwargs["compressors"] = byte_compressors
    return codec_kwargs
