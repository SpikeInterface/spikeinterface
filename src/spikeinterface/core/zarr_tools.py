import json

import numpy as np


def write_json_array(group, name: str, data, compressor=None):
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
    compressor : numcodecs.Codec | None, default: None
        The compressor of the array.

    Returns
    -------
    array : zarr.Array
        The created array.
    """
    payload = np.frombuffer(json.dumps(data).encode("utf-8"), dtype="uint8")
    array = group.create_dataset(name=name, data=payload, compressor=compressor)
    array.attrs["encoding"] = "json"
    return array


def read_encoded_array(array):
    """
    Read data saved by `write_json_array`, or a legacy length-1 object array (json or pickle codec)
    saved by spikeinterface<=0.105.

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
    raise ValueError(
        f"Unknown encoding {encoding!r} for {array.name}. It was probably saved by a newer version of spikeinterface."
    )
