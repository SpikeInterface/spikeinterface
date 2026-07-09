"""
Canonical serializable objects for cross-version serialization testing.

This module is imported and executed under OLD spikeinterface installs to generate
fixtures, so the builders may use ONLY public API that existed in the oldest version
in the test matrix. Keep them on the most stable public surface, and import moved
classes portably (try the new location, fall back to the old).

The invariant checks run under the CURRENT version in the loader test and may use
anything.

Two axes are covered, chosen per entry via "formats":
  - "json" / "pickle": dump_to_json / dump_to_pickle store only the recipe (class path
    + kwargs). Both reload through the same from_dict path. Targets the moved-class and
    changed-signature axes. Does NOT carry properties/annotations.
  - "binary" / "zarr" (recordings) and "numpy_folder" (sortings): save() materializes
    the full state (traces or spike trains, properties, annotations, probe) to disk.
    Targets the on-disk encoding axis: property and annotation preservation, and the
    probe representation. "binary" is recording-only, hence "numpy_folder" for sortings.
"""
from packaging.version import parse
from spikeinterface import __version__ as si_version


# Filename suffix per format, relative to the fixtures dir (folder formats use a suffix
# rather than an extension). Both the generator and the loader build a fixture path as
# f"{entry_id}{FIXTURE_SUFFIX[fmt]}", so the writer and reader agree by construction.
FIXTURE_SUFFIX = {
    "json": ".json",
    "pickle": ".pkl",
    "binary": "_binary",
    "numpy_folder": "_numpy_folder",
    "zarr": ".zarr",
}


# --- json (recipe) entries: moved class + a second class -------------------------


def _build_noise_generator_recording():
    # NoiseGeneratorRecording moved from spikeinterface.core.generate to
    # spikeinterface.generation. Import portably; the serialized "class" path is
    # whatever the running version records, which is what exercises the redirect.
    try:
        from spikeinterface.generation import NoiseGeneratorRecording
    except ImportError:
        from spikeinterface.core.generate import NoiseGeneratorRecording

    return NoiseGeneratorRecording(num_channels=4, sampling_frequency=30000.0, durations=[1.0, 1.5], seed=0)


def _check_noise_generator_recording(rec):
    assert type(rec).__name__ == "NoiseGeneratorRecording", type(rec).__name__
    assert rec.get_num_channels() == 4
    assert rec.get_num_segments() == 2
    assert rec.get_sampling_frequency() == 30000.0


def _build_mock_recording():
    from spikeinterface.core import generate_recording

    return generate_recording(num_channels=4, durations=[1.0], sampling_frequency=30000.0, seed=0)


def _check_mock_recording(rec):
    from spikeinterface.core import BaseRecording

    assert isinstance(rec, BaseRecording), type(rec).__name__
    assert rec.get_num_channels() == 4
    assert rec.get_num_segments() == 1


# --- binary/zarr/numpy_folder (materialized) entries: encoding axis --------------


def _build_recording_with_properties():
    import numpy as np
    from spikeinterface.core import generate_recording

    rec = generate_recording(num_channels=4, durations=[1.0], sampling_frequency=30000.0, seed=0)
    rec.set_property("quality", np.array(["good", "good", "bad", "good"]))
    rec.annotate(experimenter="test")
    return rec


def _check_recording_with_properties(rec):
    assert rec.get_num_channels() == 4
    assert list(rec.get_property("quality")) == ["good", "good", "bad", "good"]
    assert rec.get_annotation("experimenter") == "test"


def _build_recording_with_probe():
    import numpy as np
    from probeinterface import generate_linear_probe
    from spikeinterface.core import generate_recording

    rec = generate_recording(num_channels=8, durations=[1.0], sampling_frequency=30000.0, seed=0)
    probe = generate_linear_probe(num_elec=8)
    probe.set_device_channel_indices(np.arange(8))
    if parse(si_version) <= parse("0.105.0"):
        rec_with_probe = rec.set_probe(probe, in_place=False)  # old API returns a new recording; portable across versions
    else:
        rec_with_probe = rec.set_probe(probe)  # new API returns a new recording; portable across versions
    return rec_with_probe


def _check_recording_with_probe(rec):
    import numpy as np

    assert rec.get_num_channels() == 8
    assert rec.has_probe()
    # Assert exact per-channel positions, not just shape: a shape-only check would pass
    # through a silent channel-to-location scramble, which is exactly the encoding-change
    # failure mode we care about.
    expected = np.array([[0.0, float(y)] for y in range(0, 160, 20)])
    assert np.array_equal(rec.get_channel_locations(), expected), rec.get_channel_locations().tolist()


def _build_recording_with_interleaved_probes():
    import numpy as np
    from probeinterface import ProbeGroup, generate_linear_probe
    from spikeinterface.core import generate_recording

    rec = generate_recording(num_channels=8, durations=[1.0], sampling_frequency=30000.0, seed=0)
    probe0 = generate_linear_probe(num_elec=4)
    probe1 = generate_linear_probe(num_elec=4)
    probe1.move([100.0, 0.0])
    probegroup = ProbeGroup()
    probegroup.add_probe(probe0)
    probegroup.add_probe(probe1)
    probegroup.set_global_device_channel_indices([0, 2, 4, 6, 1, 3, 5, 7])
    # Interleave the two probes' channels: channel i alternates between probe0 and probe1.
    if parse(si_version) <= parse("0.105.0"):
        rec_with_probe = rec.set_probegroup(probegroup, in_place=False)  # old API returns a new recording; portable across versions
    else:
        rec_with_probe = rec.set_probegroup(probegroup)  # new API returns a new recording; portable across versions
    return rec_with_probe


def _check_recording_with_interleaved_probes(rec):
    import numpy as np

    assert rec.get_num_channels() == 8
    assert rec.has_probe()
    # The interleaved channel-to-contact mapping must survive the round-trip. This is the
    # multi-probe scramble case (the one that motivated _global_contact_order in #4465):
    # if the encoding change mis-orders contacts, these exact positions will not match.
    expected = np.array(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [0.0, 20.0],
            [100.0, 20.0],
            [0.0, 40.0],
            [100.0, 40.0],
            [0.0, 60.0],
            [100.0, 60.0],
        ]
    )
    assert np.array_equal(rec.get_channel_locations(), expected), rec.get_channel_locations().tolist()


def _build_preprocessed_chain():
    from spikeinterface.core import generate_recording
    from spikeinterface.preprocessing import common_reference, scale

    rec = generate_recording(num_channels=4, durations=[1.0], sampling_frequency=30000.0, seed=0)
    # Two nested scipy-free preprocessing wrappers (scale then common_reference): this
    # exercises recursive parent reload without pulling scipy into the environments.
    return common_reference(scale(rec, gain=2.0))


def _check_preprocessed_chain(rec):
    # The outer wrapper and the recursive parent chain must both reload (the kwargs
    # embed the parent recording dict, so this exercises recursive deserialization).
    assert type(rec).__name__ == "CommonReferenceRecording", type(rec).__name__
    assert rec.get_num_channels() == 4
    assert rec.get_num_segments() == 1


def _build_sorting():
    from spikeinterface.core import generate_sorting

    return generate_sorting(num_units=5, sampling_frequency=30000.0, durations=[1.0])


def _check_sorting(sorting):
    assert sorting.get_num_units() == 5, sorting.get_num_units()
    assert sorting.get_num_segments() == 1
    spike_train = sorting.get_unit_spike_train(sorting.unit_ids[0], segment_index=0)
    assert spike_train.ndim == 1


def _build_sorting_with_properties():
    import numpy as np
    from spikeinterface.core import generate_sorting

    sorting = generate_sorting(num_units=4, sampling_frequency=30000.0, durations=[1.0])
    sorting.set_property("quality", np.array(["good", "good", "bad", "good"]))
    sorting.annotate(experimenter="test")
    return sorting


def _check_sorting_with_properties(sorting):
    assert sorting.get_num_units() == 4
    assert list(sorting.get_property("quality")) == ["good", "good", "bad", "good"]
    assert sorting.get_annotation("experimenter") == "test"


OBJECTS = [
    {
        "id": "noise_generator_recording",
        "build": _build_noise_generator_recording,
        "check": _check_noise_generator_recording,
        "formats": ["json", "pickle"],
    },
    {
        "id": "mock_recording",
        "build": _build_mock_recording,
        "check": _check_mock_recording,
        "formats": ["json", "pickle"],
    },
    {
        "id": "recording_with_properties",
        "build": _build_recording_with_properties,
        "check": _check_recording_with_properties,
        "formats": ["binary", "zarr"],
    },
    {
        "id": "recording_with_probe",
        "build": _build_recording_with_probe,
        "check": _check_recording_with_probe,
        "formats": ["binary", "zarr"],
    },
    {
        "id": "recording_with_interleaved_probes",
        "build": _build_recording_with_interleaved_probes,
        "check": _check_recording_with_interleaved_probes,
        "formats": ["binary", "zarr"],
    },
    {
        "id": "preprocessed_chain",
        "build": _build_preprocessed_chain,
        "check": _check_preprocessed_chain,
        "formats": ["json", "pickle"],
    },
    {
        "id": "sorting",
        "build": _build_sorting,
        "check": _check_sorting,
        "formats": ["numpy_folder", "zarr"],
    },
    {
        "id": "sorting_with_properties",
        "build": _build_sorting_with_properties,
        "check": _check_sorting_with_properties,
        "formats": ["numpy_folder", "zarr"],
    },
]
