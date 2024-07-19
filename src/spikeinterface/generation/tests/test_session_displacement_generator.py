import pytest

from spikeinterface.generation.session_displacement_generator import generate_inter_session_displacement_recordings
from spikeinterface.generation.drifting_generator import generate_drifting_recording
from spikeinterface.core import order_channels_by_depth
import numpy as np
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks


class TestSessionDisplacementGenerator:

    @pytest.fixture(scope="session")
    def displaced_recording(self):

        info = {
            "kwargs": {
                "rec_durations": (10, 10, 25, 33),
                "rec_shifts": ((0, 0), (2, -100), (-3, 275), (4, 1e6)),
                "num_units": 5,
                "seed": 42,
            },
            "num_recs": 4,
            "y_bin_um": 10,
        }
        info["kwargs"]["generate_probe_kwargs"] = dict(
            num_columns=1,
            num_contact_per_column=128,
            xpitch=16,
            ypitch=info["y_bin_um"],
            contact_shapes="square",
            contact_shape_params={"width": 12},
        )

        output_recordings, output_sorting, extra_outputs = (
            generate_inter_session_displacement_recordings(  # TODO; fixture
                **info["kwargs"],
                extra_outputs=True,
            )
        )

        return output_recordings, output_sorting, extra_outputs, info

    def get_peak_chan_loc_in_um(self, template_array, y_bin_um):
        return np.argmax(np.max(template_array, axis=0)) * y_bin_um

    def test_x_y_rigid_shifts_are_properly_set(self, displaced_recording):
        """ """
        output_recordings, _, extra_outputs, info = displaced_recording
        num_units = info["kwargs"]["num_units"]
        rec_shifts = info["kwargs"]["rec_shifts"]

        # test unit locations are shifted as expected according
        # to the record shifts
        locations_1 = extra_outputs["unit_locations"][0]

        for rec_idx in range(1, 4):

            shifts = rec_shifts[rec_idx]

            assert np.array_equal(
                locations_1 + np.r_[shifts, 0].astype(np.float32), extra_outputs["unit_locations"][rec_idx]
            )

        # Check that the generated templates are correctly shifted
        # For each generated unit, check that the max loading channel is
        # shifted as expected. In the case that the unit location is off the
        # probe, check the maximum lowest channel is the min / max channel on
        # the probe, or zero (the unit is too far to reach the probe).
        min_channel_loc = output_recordings[0].get_channel_locations()[0, 1]
        max_channel_loc = output_recordings[0].get_channel_locations()[-1, 1]
        for unit_idx in range(num_units):

            start_pos = self.get_peak_chan_loc_in_um(
                extra_outputs["template_array_moved"][0][unit_idx],
                info["y_bin_um"],
            )

            for rec_idx in range(1, info["num_recs"]):

                new_pos = self.get_peak_chan_loc_in_um(
                    extra_outputs["template_array_moved"][rec_idx][unit_idx], info["y_bin_um"]
                )

                y_shift = rec_shifts[rec_idx][1]
                if start_pos + y_shift > max_channel_loc:
                    assert new_pos == max_channel_loc or new_pos == 0
                elif start_pos + y_shift < min_channel_loc:
                    assert new_pos == min_channel_loc or new_pos == 0
                else:
                    assert np.isclose(new_pos, start_pos + y_shift, info["y_bin_um"])

        # Confidence check the correct templates are
        # loaded to the recording object.
        for rec_idx in range(info["num_recs"]):
            assert np.array_equal(
                output_recordings[rec_idx].templates,
                extra_outputs["template_array_moved"][rec_idx],
            )
        # TODO: document about random seed behaviour, indeed spike times are differet (other things will be too) across recordings but not if the seed is fixed.

    def test_spike_times_across_recordings(self, displaced_recording):

        _, _, _, info = displaced_recording

        num_recs = info.pop("num_recs")
        info["kwargs"]["rec_durations"] = (10,) * num_recs

        _, output_sortings_same = generate_inter_session_displacement_recordings(**info["kwargs"])

        info["kwargs"]["seed"] = None
        _, output_sortings_different = generate_inter_session_displacement_recordings(**info["kwargs"])

        for unit_idx in range(info["kwargs"]["num_units"]):
            for rec_idx in range(1, num_recs):

                assert np.array_equal(
                    output_sortings_same[0].get_unit_spike_train(unit_idx),
                    output_sortings_same[rec_idx].get_unit_spike_train(unit_idx),
                )
                assert not np.array_equal(
                    output_sortings_different[0].get_unit_spike_train(unit_idx),
                    output_sortings_different[rec_idx].get_unit_spike_train(unit_idx),
                )

    @pytest.mark.parametrize("dim_idx", [0, 1])
    def test_x_y_shift_non_rigid(self, displaced_recording, dim_idx):
        """ """
        _, _, _, info = displaced_recording

        info["kwargs"]["rec_shifts"] = ((0, 0), (10, 15), (15, 20), (20, 25))

        _, _, extra_rigid = generate_inter_session_displacement_recordings(
            **info["kwargs"], non_rigid_gradient=None, extra_outputs=True
        )
        _, _, extra_nonrigid_max = generate_inter_session_displacement_recordings(
            **info["kwargs"], non_rigid_gradient=0, extra_outputs=True
        )
        _, _, extra_nonrigid_none = generate_inter_session_displacement_recordings(
            **info["kwargs"], non_rigid_gradient=1, extra_outputs=True
        )

        initial_locations = extra_rigid["unit_locations"][0]

        for rec_idx in range(1, info["num_recs"]):

            y_shifts_rigid = self.get_shifts(extra_rigid, rec_idx, dim_idx, initial_locations)
            y_shifts_rigid = np.round(y_shifts_rigid, 5)

            assert np.unique(y_shifts_rigid).size == 1

            shift = info["kwargs"]["rec_shifts"][rec_idx][dim_idx]

            y_shifts_nonrigid = self.get_shifts(extra_nonrigid_max, rec_idx, dim_idx, initial_locations)

            x = np.linalg.norm(initial_locations, axis=1)
            y = (x - np.min(x)) / (np.max(x) - np.min(x))

            assert np.unique(y_shifts_nonrigid).size == info["kwargs"]["num_units"]

            # There is some small rounding error due to difference in distance computation,
            # the main thing is the relative order not the absolute value.
            assert np.allclose(y_shifts_nonrigid, shift * y, rtol=0, atol=0.5)  # TODO@: chec kthis ther

            # then do again with non-ridig-gradient 1 and check it matches rigid case!!
            y_shifts_rigid_2 = self.get_shifts(extra_nonrigid_none, rec_idx, dim_idx, initial_locations)

            assert np.array_equal(y_shifts_rigid, np.round(y_shifts_rigid_2, 5))

    def get_shifts(self, extras_dict, rec_idx, dim_idx, initial_locations):
        return extras_dict["unit_locations"][rec_idx][:, dim_idx] - initial_locations[:, dim_idx]

    def test_x_y_shift_peak_detection(self, displaced_recording):  # TODO: test something going off the probe

        _, _, _, info = displaced_recording

        # the seed is important here, otherwise the unit positions
        # might go off the end of the probe
        y_shift = -100
        info["kwargs"]["rec_shifts"] = ((0, 0), (0, y_shift))
        info["kwargs"]["rec_durations"] = (0.5, 0.5)
        info["num_recs"] = 2
        info["kwargs"]["num_units"] = 1

        output_recordings, _, extra_rigid = generate_inter_session_displacement_recordings(
            **info["kwargs"], generate_noise_kwargs=dict(noise_levels=(1.0, 2.0), spatial_decay=1.0), extra_outputs=True
        )

        first_recording = output_recordings[0]

        peaks = detect_peaks(first_recording, method="by_channel")
        peak_locs = localize_peaks(first_recording, peaks, method="center_of_mass")
        first_pos = np.median(peak_locs["y"])

        shifted_recording = output_recordings[1]
        peaks = detect_peaks(shifted_recording, method="by_channel")
        peak_locs = localize_peaks(shifted_recording, peaks, method="center_of_mass")

        new_pos = np.median(peak_locs["y"])

        assert np.isclose(new_pos, first_pos + y_shift, rtol=0, atol=1)

    def test_same_as_generate_ground_truth_recording(self):
        """
        It is expected that inter-session displacement randomly
        generated recording and injected motion recording will
        use exactly the same method to generate the ground-truth
        recording (without displacement or motion). To check this,
        set their kwargs equal and seed, then generate a non-displaced
        recording. It should be identical to the static recroding
        generated by `generate_drifting_recording()`.
        """

        # Set some shared kwargs
        num_units = 5
        duration = 10
        sampling_frequency = 30000.0
        probe_name = "Neuropixel-128"
        generate_probe_kwargs = None
        generate_unit_locations_kwargs = dict()
        generate_templates_kwargs = dict(ms_before=1.5, ms_after=3)
        generate_sorting_kwargs = dict()
        generate_noise_kwargs = dict()
        seed = 42

        # Generate a inter-session displacement recording with no displacement
        no_shift_recording, _ = generate_inter_session_displacement_recordings(
            num_units=num_units,
            rec_durations=[duration],
            rec_shifts=((0, 0)),
            sampling_frequency=sampling_frequency,
            probe_name=probe_name,
            generate_probe_kwargs=generate_probe_kwargs,
            generate_unit_locations_kwargs=generate_unit_locations_kwargs,
            generate_templates_kwargs=generate_templates_kwargs,
            generate_sorting_kwargs=generate_sorting_kwargs,
            generate_noise_kwargs=generate_noise_kwargs,
            seed=seed,
        )
        no_shift_recording = no_shift_recording[0]

        # Generate a drifting recording with no drift
        static_recording, _, _ = generate_drifting_recording(
            num_units=num_units,
            duration=duration,
            sampling_frequency=sampling_frequency,
            probe_name=probe_name,
            generate_probe_kwargs=generate_probe_kwargs,
            generate_unit_locations_kwargs=generate_unit_locations_kwargs,
            generate_templates_kwargs=generate_templates_kwargs,
            generate_sorting_kwargs=generate_sorting_kwargs,
            generate_noise_kwargs=generate_noise_kwargs,
            generate_displacement_vector_kwargs=dict(
                motion_list=[
                    dict(
                        drift_mode="zigzag",
                        non_rigid_gradient=None,
                        t_start_drift=1.0,
                        t_end_drift=None,
                        period_s=200,
                    ),
                ]
            ),
            seed=seed,
        )

        # Check the templates and raw data match exactly.
        assert np.array_equal(
            no_shift_recording.get_traces(start_frame=0, end_frame=10),
            static_recording.get_traces(start_frame=0, end_frame=10),
        )

        assert np.array_equal(no_shift_recording.templates, static_recording.drifting_templates.templates_array)
