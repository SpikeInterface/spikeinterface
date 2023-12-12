import numpy as np
from spikeinterface.core import Templates


def interpolate_templates(templates_array, source_locations, dest_locations, interpolation_method="cubic"):
    """
    Interpolate templates_array to new positions.
    Useful to create motion or to remap templates_array form probeA to probeB.

    Note that several moves can be done by broadcasting when dest_locations have more than one dimension.
    
    Parameters
    ----------
    templates_array: np.array
        A numpy array with dense templates_array.
        shape = (num_template, num_sample, num_channel)
    source_locations: np.array 
        The channel source location corresponding to templates_array.
        shape = (num_channel, 2)
    dest_locations: np.array
        The new channel position, if ndim == 3, then the interpolation is broadcated with last dim.
        shape = (num_channel, 2) or (num_motion, num_channel, 2)
    interpolation_method: str, default "cubic"
        The interpolation method.
    
    Returns
    -------
    new_templates_array: np.array
        shape = (num_template, num_sample, num_channel) or = (num_motion, num_template, num_sample, num_channel, )
    """
    import scipy.interpolate

    source_locations = np.asarray(source_locations)
    dest_locations = np.asarray(dest_locations)
    if dest_locations.ndim == 2:
        new_shape = templates_array.shape
    elif dest_locations.ndim == 3:
        new_shape = (dest_locations.shape[0], ) + templates_array.shape
    else:
        raise ValueError(f"Incorrect dimensions for dest_locations: {dest_locations.ndim}. Dimension can be 2 or 3. ")

    new_templates_array = np.zeros(new_shape, dtype=templates_array.dtype)
    
    for template_ind in range(templates_array.shape[0]):
        for time_ind in range(templates_array.shape[1]):
            template = templates_array[template_ind, time_ind, :]
            interp_template = scipy.interpolate.griddata(source_locations, template, dest_locations,
                                                         method=interpolation_method,
                                                         fill_value=0)
            if dest_locations.ndim == 2:
                new_templates_array[template_ind, time_ind, :] = interp_template
            elif dest_locations.ndim == 3: 
                new_templates_array[:, template_ind, time_ind, :] = interp_template

    return new_templates_array




def move_dense_templates(templates_array, displacements, source_probe, dest_probe=None, interpolation_method="cubic"):
    """
    Move all templates_array given some displacements using spatial interpolation (cubic or linear).
    Optionally can be remapped to another probe with a different geometry.
    
    This function operates on dense template.
    
    Note : in this function no checks are done to see if templates_array can be interpolatable after displacements.
    To check if the given displacements are interpolatable use the higher level function move_templates().
    
    Parameters
    ----------
    templates_array: np.array
        A numpy array with dense templates_array.
        shape = (num_template, num_sample, num_channel)
    displacements: np.array
        Displacement vector
        shape: (num_displacement, 2)
    source_probe: Probe
        The Probe object on which templates_array are defined
    dest_probe: Probe | None, default: None
        The destination Probe. Can be different geometry than the original.
        If None then the same probe  is used.
    interpolation_method: "cubic" | "linear", default: "cubic"
        The interpolation method.        
    
    Returns
    -------
    new_templates_array: np.array
        shape = (num_displacement, num_template, num_sample, num_channel, )
    """
    assert displacements.ndim == 2
    assert displacements.shape[1] == 2

    if dest_probe is None:
        dest_probe = source_probe
    src_channel_locations = source_probe.contact_positions
    dest_channel_locations = dest_probe.contact_positions
    moved_locations = dest_channel_locations[np.newaxis, :, :] - displacements.reshape(-1, 1, 2)
    templates_array_moved = interpolate_templates(templates_array, src_channel_locations, moved_locations, interpolation_method=interpolation_method)
    return templates_array_moved






def move_templates(templates, displacements, source_probe, dest_probe=None, interpolation_method="cubic"):
    """
    Move all templates given some displacements using spatial interpolation (cubic or linear).
    Optionally can be remapped to another probe with a different geometry.

    Before interpolation this function checks that the templates can be interpolated to new
    positions using the sparsity information: the entire sparse templates must be covered at all new positions.
    If the Templates object is dense then no check is possible and a warning is emitted.

    Parameters
    ----------
    templates: Templates
        A numpy array with dense templates.
        shape = (num_template, num_sample, num_channel)
    displacements: np.array
        Displacement vector
        shape: (num_displacement, 2)
    source_probe: Probe
        The Probe object on which templates are defined
    dest_probe: Probe or None
        The destination Probe. Can be different geometry than the original.
        If None then the same probe  is used.
    interpolation_method: "cubic" | "linear", default: "cubic"
        The interpolation method.
    
    Returns
    -------
    new_templates: list of Templates

    """
    if templates.are_templates_sparse():
        # TODO make some check!!
        pass
    else:
        warnings.warn("move_templates() with dense templates cannot checks that template can be interpolated at all places")
    
    dense_templates = templates.get_dense_templates()
    dense_templates_moved = move_dense_templates(templates, displacements, source_probe, dest_probe=None, interpolation_method="cubic")

    new_templates = []
    for i in range(dense_templates_moved.shape[0]):
        dense_templates = Templates(
            templates_array=dense_templates_moved,
            sampling_frequency=templates.sampling_frequency,
            nbefore=templates.nbefore,
            channel_ids=templates.channel_ids,
            unit_ids=templates.unit_ids,
        )
        # TODO : sparsify back the templates
        new_templates.append(dense_templates)

    return new_templates





