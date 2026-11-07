from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.postprocessing import ComputeUnitLocations
import pytest
from probeinterface import Probe
from spikeinterface.core import create_sorting_analyzer, generate_ground_truth_recording
from spikeinterface.postprocessing.localization_tools import (
    estimate_distance_error_with_log,
    make_initial_guess_and_bounds,
    solve_monopolar_triangulation,
)
import numpy as np


def test_monopolar_log_finite_differences(monkeypatch):
    from scipy.optimize import minimize

    # An ambiguous real waveform has two local minima. Changing the numerical
    # gradient can select the worse one, so preserve the original solver path.
    amplitudes = [4, 7, 6, 7, 3, 10, 33, 8, 7, 9, 8, 35]
    contact_locations = np.array(
        [
            [32, 3100],
            [16, 3120],
            [48, 3120],
            [0, 3140],
            [32, 3140],
            [16, 3160],
            [48, 3160],
            [0, 3180],
            [32, 3180],
            [16, 3200],
            [48, 3200],
            [32, 3220],
        ],
        dtype="float64",
    )
    for dtype, max_distance, function_limit in [
        ("int16", 150, 15000),
        ("float32", 150, 15000),
        ("float64", 150, 15000),
        ("float64", 2, 15000),
        ("float64", 0.1, 15000),
        ("float64", 150, 4),
    ]:
        waveform_data = np.array(amplitudes, dtype=dtype)
        initial, bounds = make_initial_guess_and_bounds(waveform_data, contact_locations, max_distance)
        bounds = list(zip(bounds[0][:3], bounds[1][:3]))
        args = (waveform_data, contact_locations, waveform_data.max())
        expected = minimize(
            estimate_distance_error_with_log,
            initial[:3],
            bounds=bounds,
            args=args,
            options={"maxfun": function_limit},
        )
        calls = []

        def minimize_and_record(*args, **kwargs):
            calls.append(kwargs.get("jac", False))
            if kwargs.get("jac"):
                # Also exercise budget exhaustion without thousands of evaluations.
                assert kwargs["options"]["maxfun"] * 4 == 15000
                kwargs["options"]["maxfun"] = function_limit // 4
            else:
                kwargs["options"] = {"maxfun": function_limit}
            result = minimize(*args, **kwargs)
            if kwargs.get("jac"):
                assert result.nfev == result.njev
            assert result.status == expected.status
            return result

        monkeypatch.setattr("scipy.optimize.minimize", minimize_and_record)
        actual = solve_monopolar_triangulation(
            waveform_data, contact_locations, max_distance, "minimize_with_log_penality"
        )
        assert calls == [max_distance > 0.1]
        np.testing.assert_array_equal(np.asarray(actual[:3]).view("uint64"), expected.x.view("uint64"))
        assert np.all(np.isfinite(actual))


class TestUnitLocationsExtension(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize(
        "params",
        [
            dict(method="center_of_mass", radius_um=100),
            dict(method="grid_convolution", radius_um=50),
            dict(method="grid_convolution", radius_um=150, weight_method={"mode": "gaussian_2d"}),
            dict(method="monopolar_triangulation", radius_um=150),
            dict(method="monopolar_triangulation", radius_um=150, optimizer="minimize_with_log_penality"),
            dict(method="max_channel"),
        ],
    )
    def test_extension(self, params):
        self.run_extension_tests(ComputeUnitLocations, params=params)


def test_2d_and_3d_unit_localization():
    """
    Our localization tools do not use the 3rd dimension of contact position.
    Hence if we pass the same data with a 2D probe and a 3D probe (with the
    same 2D positions), we should get the same result for all methods.

    Also serves as an integration test of all the localization methods for
    2D and 3D channel locations.
    """

    # make a 3D synthetic recording
    positions_3D = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])

    probe = Probe(ndim=3, si_units="um")
    probe.set_contacts(positions=positions_3D, plane_axes=[[[1, 0, 0], [0, 1, 0]] * 4])

    probe.set_device_channel_indices(np.arange(4))

    recording_3D, sorting_3D = generate_ground_truth_recording(num_channels=4, num_units=2, probe=probe, seed=1205)
    analyzer_3D = create_sorting_analyzer(sorting_3D, recording_3D, sparse=False)
    analyzer_3D.compute(["random_spikes", "templates"])

    # make a 2D synthetic recording
    recording_2D = recording_3D

    positions_2D = positions_3D[:, 0:2]
    probe_2D = Probe(ndim=2, si_units="um")
    probe_2D.set_contacts(positions=positions_2D)
    probe_2D.set_device_channel_indices(np.arange(4))
    recording_2D.set_probe(probe=probe_2D)

    sorting_2D = sorting_3D

    analyzer_2D = create_sorting_analyzer(sorting_2D, recording_2D, sparse=False)
    analyzer_2D.compute(["random_spikes", "templates"])

    for method in ["center_of_mass", "grid_convolution", "monopolar_triangulation", "max_channel"]:

        analyzer_2D.compute("unit_locations", method=method)
        analyzer_3D.compute("unit_locations", method=method)
        unit_locations_2D = analyzer_2D.get_extension("unit_locations").get_data()
        unit_locations_3D = analyzer_3D.get_extension("unit_locations").get_data()

        assert np.all(unit_locations_2D[:, :2] == unit_locations_3D[:, :2])
