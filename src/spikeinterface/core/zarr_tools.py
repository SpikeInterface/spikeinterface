import warnings

import numpy as np
import zarr

# metadata keys that are not members of a group (zarr v2 and v3)
_ZARR_METADATA_KEYS = (".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json")


def check_compressors_match(comp1, comp2, skip_typesize=True):
    """
    Check if two compressor objects match.

    Parameters
    ----------
    comp1 : zarr.Codec | Tuple[zarr.Codec]
        The first compressor object to compare.
    comp2 : zarr.Codec | Tuple[zarr.Codec]
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


def is_sklearn_estimator(obj) -> bool:
    """
    Check whether an object looks like a fitted scikit-learn estimator
    (e.g. the PCA models of the "principal_components" extension).
    """
    return (
        callable(getattr(obj, "get_params", None))
        and callable(getattr(obj, "set_params", None))
        and hasattr(obj, "__dict__")
    )


def save_sklearn_model_to_zarr_group(parent_group, name: str, model, **saving_options) -> None:
    """
    Save a scikit-learn estimator in a zarr sub-group without using pickle.

    The state of the estimator (`vars(model)`) is split in two: numpy arrays are saved as
    zarr arrays and the remaining (json-serializable) values are saved in the
    "sklearn_model" attribute of the group, together with the class of the estimator.
    See `load_sklearn_model_from_zarr_group` for the reverse operation.

    Parameters
    ----------
    parent_group : zarr.Group
        The zarr group in which the sub-group is created.
    name : str
        The name of the sub-group.
    model : sklearn estimator
        The estimator to save.
    **saving_options : dict
        Options passed to `zarr.Group.create_array` for the array parts of the state.

    Raises
    ------
    ValueError
        If part of the state of the estimator can neither be saved as a zarr array nor
        serialized to json (e.g. a nested estimator or an arbitrary python object).
    """
    import json

    state = {}
    arrays = {}
    for key, value in vars(model).items():
        if isinstance(value, np.ndarray):
            if value.dtype.kind == "O":
                raise ValueError(f"Cannot save object-dtype array '{key}' of {type(model).__name__} to zarr")
            arrays[key] = value
        else:
            if isinstance(value, np.generic):
                value = value.item()
            try:
                json.dumps(value)
            except TypeError:
                raise ValueError(
                    f"Cannot save attribute '{key}' of {type(model).__name__} to zarr: "
                    f"{type(value).__name__} is not json-serializable"
                )
            state[key] = value

    model_group = parent_group.create_group(name)
    for key, value in arrays.items():
        model_group.create_array(name=key, data=value, **saving_options)

    model_info = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "state": state,
        "array_keys": sorted(arrays.keys()),
    }
    try:
        import sklearn

        model_info["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass
    model_group.attrs["sklearn_model"] = model_info


def load_sklearn_model_from_zarr_group(model_group):
    """
    Rebuild a scikit-learn estimator saved by `save_sklearn_model_to_zarr_group`.

    Parameters
    ----------
    model_group : zarr.Group
        The zarr group containing the estimator state.

    Returns
    -------
    model : sklearn estimator
        The estimator, with the same state it had when saved.
    """
    import importlib

    model_info = model_group.attrs["sklearn_model"]
    module_name, _, class_name = model_info["class"].rpartition(".")
    model_class = getattr(importlib.import_module(module_name), class_name)

    # the full state is restored, so the constructor is bypassed on purpose
    model = model_class.__new__(model_class)
    for key, value in model_info["state"].items():
        setattr(model, key, value)
    for key in model_info["array_keys"]:
        setattr(model, key, np.asarray(model_group[key][...]))

    return model
