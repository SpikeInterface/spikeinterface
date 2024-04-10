import pytest
import numpy as np
from pathlib import Path
import shutil

from spikeinterface.generation import (
    interpolate_templates,
    move_dense_templates,
    make_linear_displacement,
    DriftingTemplates,
    InjectDriftingTemplatesRecording,
)
from spikeinterface.core.generate import generate_templates, generate_sorting, NoiseGeneratorRecording
from spikeinterface.core import Templates, BaseRecording

from probeinterface import generate_multi_columns_probe


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "generation"
else:
    cache_folder = Path("cache_folder") / "generation"


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
    probe.set_device_channel_indices(np.arange(probe.contact_ids.size))

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

    sampling_frequency = 30000.0
    ms_before = 1.0
    ms_after = 3.0

    nbefore = int(sampling_frequency * ms_before)

    generate_kwargs = dict(
        sampling_frequency=sampling_frequency,
        ms_before=ms_before,
        ms_after=ms_after,
        seed=2205,
        unit_params=dict(
            alpha=(4_000.0, 8_000.0),
            depolarization_ms=(0.09, 0.16),
            spatial_decay=np.ones(num_units) * 35,
            repolarization_ms=np.ones(num_units) * 0.8,
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
    dest_locations = source_locations + np.array([2.0, 3])
    interpolate_templates(templates.templates_array, source_locations, dest_locations, interpolation_method="cubic")


def test_move_dense_templates():
    templates = make_some_templates()

    num_move = 5
    amplitude_motion_um = 20
    displacements = np.zeros((num_move, 2))
    displacements[:, 1] = np.linspace(-amplitude_motion_um, amplitude_motion_um, num_move)

    templates_moved = move_dense_templates(templates.templates_array, displacements, templates.probe)
    assert templates_moved.shape == (num_move,) + templates.templates_array.shape


def test_DriftingTemplates():
    static_templates = make_some_templates()
    drifting_templates = DriftingTemplates.from_static(static_templates)

    displacement = np.array([[5.0, 10.0]])
    unit_index = 0
    moved_template_array = drifting_templates.move_one_template(unit_index, displacement)

    num_move = 5
    amplitude_motion_um = 20
    displacements = np.zeros((num_move, 2))
    displacements[:, 1] = np.linspace(-amplitude_motion_um, amplitude_motion_um, num_move)
    drifting_templates.precompute_displacements(displacements)
    assert drifting_templates.templates_array_moved.shape == (
        num_move,
        static_templates.num_units,
        static_templates.num_samples,
        static_templates.num_channels,
    )


def test_InjectDriftingTemplatesRecording():
    templates = make_some_templates()
    probe = templates.probe

    # drifting templates
    drifting_templates = DriftingTemplates.from_static(templates)
    channel_locations = probe.contact_positions

    num_units = templates.unit_ids.size
    sampling_frequency = templates.sampling_frequency

    # spikes
    duration = 125.5
    sorting = generate_sorting(
        num_units=num_units,
        sampling_frequency=sampling_frequency,
        durations=[
            duration,
        ],
        firing_rates=25.0,
    )

    # displacement vectors
    displacement_sampling_frequency = 5.0
    times = np.arange(0, duration, 1 / displacement_sampling_frequency)

    num_motion = 29

    # 2 drifts signal with diffarents factor for units
    start = np.array([0, -15.0])
    stop = np.array([0, 12])
    mid = (start + stop) / 2
    freq0 = 0.1
    displacement_vector0 = np.sin(2 * np.pi * freq0 * times)[:, np.newaxis] * (start - stop) + mid
    freq1 = 0.01
    displacement_vector1 = 0.2 * np.sin(2 * np.pi * freq1 * times)[:, np.newaxis] * (start - stop) + mid
    displacement_vectors = np.stack([displacement_vector0, displacement_vector1], axis=2)

    displacement_unit_factor = np.zeros((num_units, 2))
    displacement_unit_factor[:, 0] = np.linspace(0.7, 0.9, num_units)
    displacement_unit_factor[:, 1] = 0.1

    # precompute discplacements
    displacements = make_linear_displacement(start, stop, num_step=num_motion)
    drifting_templates.precompute_displacements(displacements)

    # recordings
    noise = NoiseGeneratorRecording(
        num_channels=probe.contact_ids.size,
        sampling_frequency=sampling_frequency,
        durations=[duration],
        noise_levels=1.0,
        dtype="float32",
    )

    rec = InjectDriftingTemplatesRecording(
        sorting=sorting,
        parent_recording=noise,
        drifting_templates=drifting_templates,
        displacement_vectors=[displacement_vectors],
        displacement_sampling_frequency=displacement_sampling_frequency,
        displacement_unit_factor=displacement_unit_factor,
        num_samples=[int(duration * sampling_frequency)],
        amplitude_factor=None,
    )

    # check serialibility
    rec = BaseRecording.from_dict(rec.to_dict())
    print(rec)

    rec_folder = cache_folder / "rec"
    if rec_folder.exists():
        shutil.rmtree(rec_folder)
    rec.save(folder=rec_folder, n_jobs=1)


if __name__ == "__main__":
    test_interpolate_templates()
    test_move_dense_templates()
    test_DriftingTemplates()
    test_InjectDriftingTemplatesRecording()
