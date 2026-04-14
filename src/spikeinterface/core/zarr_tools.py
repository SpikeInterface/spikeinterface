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
