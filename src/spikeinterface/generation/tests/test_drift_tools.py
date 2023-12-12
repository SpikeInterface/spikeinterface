import pytest
import numpy as np

from spikeinterface.generation import interpolate_templates, move_dense_templates, DriftingTemplates
from spikeinterface.core.generate import generate_templates
from spikeinterface.core import Templates


from probeinterface import generate_multi_columns_probe




def make_some_templates():
    probe = generate_multi_columns_probe(
        num_columns=12,
        num_contact_per_column=12,
        xpitch=20,
        ypitch=20,
        # y_shift_per_column=[0, -10, 0],
        contact_shapes="square",
        contact_shape_params={"width": 10},
    )

    # import matplotlib.pyplot as plt
    # from probeinterface.plotting import plot_probe
    # plot_probe(probe)
    # plt.show()



    channel_locations = probe.contact_positions
    unit_locations = np.array(
        [
            [102, 103, 20],
            [182, 33, 20],
        ]
    )
    num_units = unit_locations.shape[0]

    sampling_frequency = 30000.
    ms_before = 1.
    ms_after = 3.

    nbefore = int(sampling_frequency * ms_before)

    generate_kwargs = dict(
        sampling_frequency=sampling_frequency,
        ms_before=ms_before,
        ms_after=ms_after,
        seed=2205,
        unit_params=dict(
            decay_power=np.ones(num_units) * 2,
            repolarization_ms=np.ones(num_units) * 0.8,
        ),
        unit_params_range=dict(
            alpha=(4_000., 8_000.),
            depolarization_ms=(0.09, 0.16),

        ),


    )
    templates_array = generate_templates(channel_locations, unit_locations, **generate_kwargs)

    templates = Templates(
        templates_array=templates_array,
        sampling_frequency=sampling_frequency,
        nbefore=nbefore,
        probe=probe,
    )


    return templates


def test_interpolate_templates():
    templates = make_some_templates()

    source_locations = templates.probe.contact_positions
    # small move on both x and y
    dest_locations = source_locations + np.array([2., 3])
    interpolate_templates(templates.templates_array, source_locations, dest_locations, interpolation_method="cubic")

def test_move_dense_templates():
    templates = make_some_templates()

    num_move = 5
    amplitude_motion_um = 20
    displacements = np.zeros((num_move, 2))
    displacements[:, 1] = np.linspace(-amplitude_motion_um, amplitude_motion_um, num_move)

    templates_moved = move_dense_templates(templates.templates_array, displacements, templates.probe)
    assert templates_moved.shape ==(num_move, ) + templates.templates_array.shape



def test_DriftingTemplates():
    static_templates = make_some_templates()
    drifting_templates = DriftingTemplates.from_static(static_templates)
    
    displacement = np.array([[5., 10.]])
    unit_index = 0
    moved_template_array = drifting_templates.move_one_template(unit_index, displacement)
    print(moved_template_array.shape)


    num_move = 5
    amplitude_motion_um = 20
    displacements = np.zeros((num_move, 2))
    displacements[:, 1] = np.linspace(-amplitude_motion_um, amplitude_motion_um, num_move)
    drifting_templates.precompute_displacements(displacements)
    assert drifting_templates.template_array_moved.shape == (num_move, static_templates.num_units, static_templates.num_samples, static_templates.num_channels)
    

if __name__ == "__main__":
    # test_interpolate_templates()
    # test_move_dense_templates()
    test_DriftingTemplates()
    
    