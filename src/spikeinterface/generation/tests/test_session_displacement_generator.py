import pytest

from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
from spikeinterface.generation.drifting_generator import generate_drifting_recording
from spikeinterface.core.generate import _ensure_firing_rates
from spikeinterface.core import order_channels_by_depth
import numpy as np
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks


class TestSessionDisplacementGenerator:
    """
    This class tests the `generate_session_displacement_recordings` that
    returns a recordings / sorting in which the units are shifted
    across sessions. This is achieved by shifting the unit locations
    in both (x, y) on the generated templates that are used in
    `InjectTemplatesRecording()`.
    """

    @pytest.fixture(scope="function")
    def options(self):
        """
        Set a set of base options that can be used in
        `generate_session_displacement_recordings() ("kwargs")
        and provide general information on the generated recordings.
        These can be edited in the tests as required.
        """
        options = {
            "kwargs": {
                "recording_durations": (10, 10, 25, 33),
                "recording_shifts": ((0, 0), (2, -100), (-3, 275), (4, 1e6)),
                "num_units": 5,
                "extra_outputs": True,
                "seed": 42,
            },
            "num_recs": 4,
            "y_bin_um": 10,
        }
        options["kwargs"]["generate_probe_kwargs"] = dict(
            num_columns=1,
            num_contact_per_column=128,
            xpitch=16,
            ypitch=options["y_bin_um"],
            contact_shapes="square",
            contact_shape_params={"width": 12},
        )

        return options

    ### Tests
    def test_x_y_rigid_shifts_are_properly_set(self, options):
        """
        The session displacement works by generating a set of
        templates shared across all recordings, but set with
        different `unit_locations()`. Check here that the
        (x, y) displacements passed in `recording_shifts` are properly
        propagated.

        First, check the set `unit_locations` are moved as expected according
        to the (x, y) shifts). Next, check the templates themselves are
        moved as expected. The x-axis shift has the effect of changing
        the template amplitude, and is not possible to test. However,
        the y-axis shift shifts the maximum signal channel, so we check
        the maximum signal channel o fthe templates is shifted as expected.
        This implicitly tests the x-axis case as if the x-axis `unit_locations`
        are shifted as expected, and the unit-locations are propagated
        to the template, then (x, y) will both be working.
        """
        output_recordings, _, extra_outputs = generate_session_displacement_recordings(**options["kwargs"])
        num_units = options["kwargs"]["num_units"]
        recording_shifts = options["kwargs"]["recording_shifts"]

        # test unit locations are shifted as expected according
        # to the record shifts
        locations_1 = extra_outputs["unit_locations"][0]

        for rec_idx in range(1, 4):

            shifts = recording_shifts[rec_idx]

            assert np.array_equal(
                locations_1 + np.r_[shifts, 0].astype(np.float32), extra_outputs["unit_locations"][rec_idx]
            )

        # Check that the generated templates are correctly shifted
        # For each generated unit, check that the max loading channel is
        # shifted as expected. In the case that the unit location is off the
        # probe, check the maximum signal channel is the min / max channel on
        # the probe, or zero (the unit is too far to reach the probe).
        min_channel_loc = output_recordings[0].get_channel_locations()[0, 1]
        max_channel_loc = output_recordings[0].get_channel_locations()[-1, 1]
        for unit_idx in range(num_units):

            start_pos = self._get_peak_chan_loc_in_um(
                extra_outputs["templates_array_moved"][0][unit_idx],
                options["y_bin_um"],
            )

            for rec_idx in range(1, options["num_recs"]):

                new_pos = self._get_peak_chan_loc_in_um(
                    extra_outputs["templates_array_moved"][rec_idx][unit_idx], options["y_bin_um"]
                )

                y_shift = recording_shifts[rec_idx][1]
                if start_pos + y_shift > max_channel_loc:
                    assert new_pos == max_channel_loc or new_pos == 0
                elif start_pos + y_shift < min_channel_loc:
                    assert new_pos == min_channel_loc or new_pos == 0
                else:
                    assert np.isclose(new_pos, start_pos + y_shift, options["y_bin_um"])

        # Confidence check the correct templates are
        # loaded to the recording object.
        for rec_idx in range(options["num_recs"]):
            assert np.array_equal(
                output_recordings[rec_idx].templates,
                extra_outputs["templates_array_moved"][rec_idx],
            )

    def _get_peak_chan_loc_in_um(self, template_array, y_bin_um):
        """
        Convenience function to get the maximally loading
        channel y-position in um for the template.
        """
        return np.argmax(np.max(template_array, axis=0)) * y_bin_um

    def test_recordings_length(self, options):
        """
        Test that the `recording_durations` that sets the
        length of each recording changes the recording
        length as expected.
        """
        output_recordings = generate_session_displacement_recordings(**options["kwargs"])[0]

        for rec, expected_rec_length in zip(output_recordings, options["kwargs"]["recording_durations"]):
            assert rec.get_total_duration() == expected_rec_length

    def test_spike_times_and_firing_rates_across_recordings(self, options):
        """
        Check the randomisation of spike times across recordings.
        When a seed is set, this is passed to `generate_sorting`
        and so the spike times across all records are expected
        to be identical. However, if no seed is set, then the spike
        times will be different across recordings.
        """
        options["kwargs"]["recording_durations"] = (10,) * options["num_recs"]

        output_sortings_same, extra_outputs_same = generate_session_displacement_recordings(**options["kwargs"])[1:3]

        options["kwargs"]["seed"] = None
        output_sortings_different, extra_outputs_different = generate_session_displacement_recordings(
            **options["kwargs"]
        )[1:3]

        for unit_idx in range(options["kwargs"]["num_units"]):
            for rec_idx in range(1, options["num_recs"]):

                # Exact spike times are not preserved when seed is None
                assert np.array_equal(
                    output_sortings_same[0].get_unit_spike_train(str(unit_idx)),
                    output_sortings_same[rec_idx].get_unit_spike_train(str(unit_idx)),
                )
                assert not np.array_equal(
                    output_sortings_different[0].get_unit_spike_train(str(unit_idx)),
                    output_sortings_different[rec_idx].get_unit_spike_train(str(unit_idx)),
                )
                # Firing rates should always be preserved.
                assert np.array_equal(
                    extra_outputs_same["firing_rates"][0][unit_idx],
                    extra_outputs_same["firing_rates"][rec_idx][unit_idx],
                )
                assert np.array_equal(
                    extra_outputs_different["firing_rates"][0][unit_idx],
                    extra_outputs_different["firing_rates"][rec_idx][unit_idx],
                )

    def test_ensure_unit_params_assumption(self):
        """
        Test the assumption that `_ensure_unit_params` does not
        change an array of firing rates, otherwise `generate_sorting`
        will internally change our firing rates.
        """
        array = np.random.randn(5)
        assert np.array_equal(_ensure_firing_rates(array, 5, None), array)

    @pytest.mark.parametrize("dim_idx", [0, 1])
    def test_x_y_shift_non_rigid(self, options, dim_idx):
        """
        Check that the non-rigid shift changes the channel location
        as expected. Non-rigid shifts are calculated depending on the
        position of the channel. The `non_rigid_gradient` parameter
        determines how much the position or 'distance' of the channel
        (w.r.t the gradient of movement) affects the scaling. When
        0, the displacement is scaled by the distance. When 0, the
        distance is ignored and all scalings are 1.

        This test checks the generated `unit_locations` under extreme
        cases, when `non_rigid_gradient` is `None` or 0, which are equivalent,
        and when it is `1`, and the displacement is directly propotional to
        the unit position.
        """
        options["kwargs"]["recording_shifts"] = ((0, 0), (10, 15), (15, 20), (20, 25))

        _, _, rigid_info = generate_session_displacement_recordings(
            **options["kwargs"],
            non_rigid_gradient=None,
        )
        _, _, nonrigid_max_info = generate_session_displacement_recordings(
            **options["kwargs"],
            non_rigid_gradient=0,
        )
        _, _, nonrigid_none_info = generate_session_displacement_recordings(
            **options["kwargs"],
            non_rigid_gradient=1,
        )

        initial_locations = rigid_info["unit_locations"][0]

        # For each recording (i.e. each recording as different displacement
        # w.r.t the first recording), check the rigid and nonrigid shifts
        # are as expected.
        for rec_idx in range(1, options["num_recs"]):

            shift = options["kwargs"]["recording_shifts"][rec_idx][dim_idx]

            # Get the rigid shift between the first recording and this shifted recording
            # Check shifts for all unit locations are all the same.
            shifts_rigid = self._get_shifts(rigid_info, rec_idx, dim_idx, initial_locations)
            shifts_rigid = np.round(shifts_rigid, 5)

            assert np.unique(shifts_rigid).size == 1

            # Get the nonrigid shift between the first recording and this recording.
            # The shift for each unit should be directly proportional to its position.
            y_shifts_nonrigid = self._get_shifts(nonrigid_max_info, rec_idx, dim_idx, initial_locations)

            distance = np.linalg.norm(initial_locations, axis=1)
            norm_distance = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))

            assert np.unique(y_shifts_nonrigid).size == options["kwargs"]["num_units"]

            # There is some small rounding error due to difference in distance computation,
            # the main thing is the relative order not the absolute value.
            assert np.allclose(y_shifts_nonrigid, shift * norm_distance, rtol=0, atol=0.5)

            # then do again with non-ridig-gradient 1 and check it matches rigid case
            shifts_rigid_2 = self._get_shifts(nonrigid_none_info, rec_idx, dim_idx, initial_locations)
            assert np.array_equal(shifts_rigid, np.round(shifts_rigid_2, 5))

    def test_non_rigid_shifts_list(self, options):
        """
        Quick check that non-rigid gradients are indeed different across
        recordings when a list of different gradients is passed.
        """
        options["kwargs"]["recording_shifts"] = ((0, 0), (0, 10), (0, 10), (0, 10))
        options["kwargs"]["seed"] = 42

        _, _, same_info = generate_session_displacement_recordings(
            **options["kwargs"],
            non_rigid_gradient=0.50,
        )
        _, _, different_info = generate_session_displacement_recordings(
            **options["kwargs"],
            non_rigid_gradient=[0.25, 0.50, 0.75, 1.0],
        )

        # Just check the first two recordings
        assert np.array_equal(same_info["unit_locations"][1], same_info["unit_locations"][2])
        assert not np.array_equal(different_info["unit_locations"][1], different_info["unit_locations"][2])

    def _get_shifts(self, extras_dict, rec_idx, dim_idx, initial_locations):
        return extras_dict["unit_locations"][rec_idx][:, dim_idx] - initial_locations[:, dim_idx]

    def test_displacement_with_peak_detection(self, options):
        """
        This test checks that the session displacement occurs
        as expected under normal usage. Create a recording with a
        single unit and a y-axis displacement. Find the peak
        locations and check the shifted peak location is as expected,
        within the tolerate of the y-axis pitch + some small error.
        """
        # The seed is important here, otherwise the unit positions
        # might go off the end of the probe. These kwargs are
        # chosen to make the recording as small as possible as this
        # test is slow for larger recordings.
        y_shift = 50
        options["kwargs"]["recording_shifts"] = ((0, 0), (0, y_shift))
        options["kwargs"]["recording_durations"] = (0.5, 0.5)
        options["num_recs"] = 2
        options["kwargs"]["num_units"] = 1
        options["kwargs"]["generate_probe_kwargs"]["num_contact_per_column"] = 18

        output_recordings, _, _ = generate_session_displacement_recordings(
            **options["kwargs"], generate_noise_kwargs=dict(noise_levels=(1.0, 2.0), spatial_decay=1.0)
        )

        first_recording = output_recordings[0]

        # Peak location of unshifted recording
        peaks = detect_peaks(first_recording, method="by_channel")
        peak_locs = localize_peaks(first_recording, peaks, method="center_of_mass")
        first_pos = np.mean(peak_locs["y"])

        # Find peak location on shifted recording and check it is
        # the original location + shift.
        shifted_recording = output_recordings[1]
        peaks = detect_peaks(shifted_recording, method="by_channel")
        peak_locs = localize_peaks(shifted_recording, peaks, method="center_of_mass")

        new_pos = np.mean(peak_locs["y"])

        # Completely arbitrary 0.5 offset to pass tests on macOS which fail around ~0.2
        # over the bin, probably due to small amount of noise.
        assert np.isclose(new_pos, first_pos + y_shift, rtol=0, atol=options["y_bin_um"] + 0.5)

    def test_amplitude_scalings(self, options):
        """
        Test that the templates are scaled by the passed scaling factors
        in the specified order. The order can be in the passed order,
        in the order of highest-to-lowest firing unit, or in the order
        of (amplitude * firing_rate) (highest to lowest unit).
        """
        # Setup arguments to create an unshifted set of recordings
        # where the templates are to be scaled with `true_scalings`
        options["kwargs"]["recording_durations"] = (10, 10)
        options["kwargs"]["recording_shifts"] = ((0, 0), (0, 0))
        options["kwargs"]["num_units"] == 5,

        true_scalings = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        recording_amplitude_scalings = {
            "method": "by_passed_order",
            "scalings": (np.ones(5), true_scalings),
        }

        _, output_sortings, extra_outputs = generate_session_displacement_recordings(
            **options["kwargs"],
            recording_amplitude_scalings=recording_amplitude_scalings,
        )

        # Check that the unit templates are scaled in the order
        # the scalings were passed.
        test_scalings = self._calculate_scalings_from_output(extra_outputs)
        assert np.allclose(test_scalings, true_scalings)

        # Now run, again applying the scalings in the order of
        # unit firing rates (highest to lowest).
        firing_rates = np.array([5, 4, 3, 2, 1])
        generate_sorting_kwargs = dict(firing_rates=firing_rates, refractory_period_ms=4.0)
        recording_amplitude_scalings["method"] = "by_firing_rate"
        _, output_sortings, extra_outputs = generate_session_displacement_recordings(
            **options["kwargs"],
            recording_amplitude_scalings=recording_amplitude_scalings,
            generate_sorting_kwargs=generate_sorting_kwargs,
        )

        test_scalings = self._calculate_scalings_from_output(extra_outputs)
        assert np.allclose(test_scalings, true_scalings[np.argsort(firing_rates)])

        # Finally, run again applying the scalings in the order of
        # unit amplitude * firing_rate
        recording_amplitude_scalings["method"] = "by_amplitude_and_firing_rate"  # TODO: method -> order
        amplitudes = np.min(np.min(extra_outputs["templates_array_moved"][0], axis=2), axis=1)
        firing_rate_by_amplitude = np.argsort(amplitudes * firing_rates)

        _, output_sortings, extra_outputs = generate_session_displacement_recordings(
            **options["kwargs"],
            recording_amplitude_scalings=recording_amplitude_scalings,
            generate_sorting_kwargs=generate_sorting_kwargs,
        )

        test_scalings = self._calculate_scalings_from_output(extra_outputs)
        assert np.allclose(test_scalings, true_scalings[firing_rate_by_amplitude])

    def _calculate_scalings_from_output(self, extra_outputs):
        first, second = extra_outputs["templates_array_moved"]
        first_min = np.min(np.min(first, axis=2), axis=1)
        second_min = np.min(np.min(second, axis=2), axis=1)
        test_scalings = second_min / first_min
        return test_scalings

    def test_metadata(self, options):
        """
        Check that metadata required to be set of generated recordings is present
        on all output recordings.
        """
        output_recordings, output_sortings, extra_outputs = generate_session_displacement_recordings(
            **options["kwargs"], generate_noise_kwargs=dict(noise_levels=(1.0, 2.0), spatial_decay=1.0)
        )
        num_chans = output_recordings[0].get_num_channels()

        for i in range(len(output_recordings)):
            assert output_recordings[i].name == "InterSessionDisplacementRecording"
            assert output_recordings[i]._annotations["is_filtered"] is True
            assert output_recordings[i].has_probe()
            assert np.array_equal(output_recordings[i].get_channel_gains(), np.ones(num_chans))
            assert np.array_equal(output_recordings[i].get_channel_offsets(), np.zeros(num_chans))

            assert np.array_equal(
                output_sortings[i].get_property("gt_unit_locations"), extra_outputs["unit_locations"][i]
            )
            assert output_sortings[i].name == "InterSessionDisplacementSorting"

    def test_shift_units_outside_probe(self, options):
        """
        When `shift_units_outside_probe` is `True`, a new set of
        units above and below the probe (y dimension) are created,
        such that they may be shifted into the recording.

        Here, check that these new units are created when `shift_units_outside_probe`
        is on and that the kwargs for the central set of units match those
        as when `shift_units_outside_probe` is `False`.
        """
        num_sessions = len(options["kwargs"]["recording_durations"])
        _, _, baseline_outputs = generate_session_displacement_recordings(
            **options["kwargs"],
        )

        _, _, outside_probe_outputs = generate_session_displacement_recordings(
            **options["kwargs"], shift_units_outside_probe=True
        )

        num_units = options["kwargs"]["num_units"]
        num_extended_units = num_units * 3

        for ses_idx in range(num_sessions):

            # There are 3x the number of units when new units are created
            # (one new set above, and one new set below the probe).
            for key in ["unit_locations", "templates_array_moved", "firing_rates"]:
                assert outside_probe_outputs[key][ses_idx].shape[0] == num_extended_units

                assert np.array_equal(
                    baseline_outputs[key][ses_idx], outside_probe_outputs[key][ses_idx][num_units:-num_units]
                )

            # The kwargs of the units in the central positions should be identical
            # to those when `shift_units_outside_probe` is `False`.
            lower_unit_pos = outside_probe_outputs["unit_locations"][ses_idx][-num_units:][:, 1]
            upper_unit_pos = outside_probe_outputs["unit_locations"][ses_idx][:num_units][:, 1]
            middle_unit_pos = baseline_outputs["unit_locations"][ses_idx][:, 1]

            assert np.min(upper_unit_pos) > np.max(middle_unit_pos)
            assert np.max(lower_unit_pos) < np.min(middle_unit_pos)

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
        probe_name = "Neuropixels1-128"
        generate_probe_kwargs = None
        generate_unit_locations_kwargs = dict()
        generate_templates_kwargs = dict(ms_before=1.5, ms_after=3)
        generate_sorting_kwargs = dict(firing_rates=1)
        generate_noise_kwargs = dict()
        seed = 42

        # Generate a inter-session displacement recording with no displacement
        no_shift_recording, _ = generate_session_displacement_recordings(
            num_units=num_units,
            recording_durations=[duration],
            recording_shifts=((0, 0),),
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
