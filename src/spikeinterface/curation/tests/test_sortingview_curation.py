import pytest
from pathlib import Path
import os
import json
import numpy as np

import spikeinterface as si
from spikeinterface.core import generate_sorting
import spikeinterface.extractors as se
from spikeinterface import set_global_tmp_folder
from spikeinterface.postprocessing import (
    compute_correlograms,
    compute_unit_locations,
    compute_template_similarity,
    compute_spike_amplitudes,
)
from spikeinterface.curation import apply_sortingview_curation

parent_folder = Path(__file__).parent
ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))
KACHERY_CLOUD_SET = bool(os.getenv("KACHERY_CLOUD_CLIENT_ID")) and bool(os.getenv("KACHERY_CLOUD_PRIVATE_KEY"))


# this needs to be run only once: if we want to regenerate we need to start with sorting result
# TODO : regenerate the
# def generate_sortingview_curation_dataset():
#     import spikeinterface.widgets as sw


#     sorting_analyzer = si.create_sorting_analyzer(sorting, recording, format="memory")
#     sorting_analyzer.compute("random_spikes")
#     sorting_analyzer.compute("waveforms")
#     sorting_analyzer.compute("templates")
#     sorting_analyzer.compute("noise_levels")
#     sorting_analyzer.compute("spike_amplitudes")
#     sorting_analyzer.compute("template_similarity")
#     sorting_analyzer.compute("unit_locations")

#     # plot_sorting_summary with curation
#     w = sw.plot_sorting_summary(sorting_analyzer, curation=True, backend="sortingview")

#     # curation_link:
#     # https://figurl.org/f?v=npm://@fi-sci/figurl-sortingview@12/dist&d=sha1://058ab901610aa9d29df565595a3cc2a81a1b08e5


@pytest.mark.skipif(ON_GITHUB and not KACHERY_CLOUD_SET, reason="Kachery cloud secrets not available")
def test_gh_curation():
    """
    Test curation using GitHub URI.
    """
    sorting = generate_sorting(num_units=10)
    # curated link:
    # https://figurl.org/f?v=npm://@fi-sci/figurl-sortingview@12/dist&d=sha1://058ab901610aa9d29df565595a3cc2a81a1b08e5
    gh_uri = "gh://SpikeInterface/spikeinterface/main/src/spikeinterface/curation/tests/sv-sorting-curation.json"
    sorting_curated_gh = apply_sortingview_curation(sorting, uri_or_json=gh_uri, verbose=True)

    assert len(sorting_curated_gh.unit_ids) == 9
    assert 1, 2 in sorting_curated_gh.unit_ids
    assert "accept" in sorting_curated_gh.get_property_keys()
    assert "mua" in sorting_curated_gh.get_property_keys()
    assert "artifact" in sorting_curated_gh.get_property_keys()

    sorting_curated_gh_accepted = apply_sortingview_curation(sorting, uri_or_json=gh_uri, include_labels=["accept"])
    sorting_curated_gh_mua = apply_sortingview_curation(sorting, uri_or_json=gh_uri, exclude_labels=["mua"])
    sorting_curated_gh_art_mua = apply_sortingview_curation(
        sorting, uri_or_json=gh_uri, exclude_labels=["artifact", "mua"]
    )
    assert len(sorting_curated_gh_accepted.unit_ids) == 3
    assert len(sorting_curated_gh_mua.unit_ids) == 6
    assert len(sorting_curated_gh_art_mua.unit_ids) == 5


@pytest.mark.skipif(ON_GITHUB and not KACHERY_CLOUD_SET, reason="Kachery cloud secrets not available")
def test_sha1_curation():
    """
    Test curation using SHA1 URI.
    """
    sorting = generate_sorting(num_units=10)

    # from SHA1
    # curated link:
    # https://figurl.org/f?v=npm://@fi-sci/figurl-sortingview@12/dist&d=sha1://058ab901610aa9d29df565595a3cc2a81a1b08e5
    sha1_uri = "sha1://449a428e8824eef9ad9bcc3241e45a2cee02d381"
    sorting_curated_sha1 = apply_sortingview_curation(sorting, uri_or_json=sha1_uri, verbose=True)
    # print(f"From SHA: {sorting_curated_sha1}")

    assert len(sorting_curated_sha1.unit_ids) == 9
    assert 1, 2 in sorting_curated_sha1.unit_ids
    assert "accept" in sorting_curated_sha1.get_property_keys()
    assert "mua" in sorting_curated_sha1.get_property_keys()
    assert "artifact" in sorting_curated_sha1.get_property_keys()
    unit_ids = sorting_curated_sha1.unit_ids
    sorting_curated_sha1_accepted = apply_sortingview_curation(sorting, uri_or_json=sha1_uri, include_labels=["accept"])
    sorting_curated_sha1_mua = apply_sortingview_curation(sorting, uri_or_json=sha1_uri, exclude_labels=["mua"])
    sorting_curated_sha1_art_mua = apply_sortingview_curation(
        sorting, uri_or_json=sha1_uri, exclude_labels=["artifact", "mua"]
    )
    assert len(sorting_curated_sha1_accepted.unit_ids) == 3
    assert len(sorting_curated_sha1_mua.unit_ids) == 6
    assert len(sorting_curated_sha1_art_mua.unit_ids) == 5


def test_json_curation():
    """
    Test curation using a JSON file.
    """
    sorting = generate_sorting(num_units=10)

    # from curation.json
    json_file = parent_folder / "sv-sorting-curation.json"
    # print(f"Sorting: {sorting.get_unit_ids()}")
    sorting_curated_json = apply_sortingview_curation(sorting, uri_or_json=json_file, verbose=True)

    assert len(sorting_curated_json.unit_ids) == 9
    assert 1, 2 in sorting_curated_json.unit_ids
    assert "accept" in sorting_curated_json.get_property_keys()
    assert "mua" in sorting_curated_json.get_property_keys()
    assert "artifact" in sorting_curated_json.get_property_keys()

    sorting_curated_json_accepted = apply_sortingview_curation(
        sorting, uri_or_json=json_file, include_labels=["accept"]
    )
    sorting_curated_json_mua = apply_sortingview_curation(sorting, uri_or_json=json_file, exclude_labels=["mua"])
    sorting_curated_json_mua1 = apply_sortingview_curation(
        sorting, uri_or_json=json_file, exclude_labels=["artifact", "mua"]
    )
    assert len(sorting_curated_json_accepted.unit_ids) == 3
    assert len(sorting_curated_json_mua.unit_ids) == 6
    assert len(sorting_curated_json_mua1.unit_ids) == 5


def test_false_positive_curation():
    """
    Test curation for false positives.
    """
    # https://spikeinterface.readthedocs.io/en/latest/modules_gallery/core/plot_2_sorting_extractor.html
    sampling_frequency = 30000.0
    duration = 20.0
    num_timepoints = int(sampling_frequency * duration)
    num_units = 20
    num_spikes = 1000
    times = np.int_(np.sort(np.random.uniform(0, num_timepoints, num_spikes)))
    labels = np.random.randint(1, num_units + 1, size=num_spikes)

    sorting = se.NumpySorting.from_times_labels(times, labels, sampling_frequency)
    # print("Sorting: {}".format(sorting.get_unit_ids()))

    json_file = parent_folder / "sv-sorting-curation-false-positive.json"
    sorting_curated_json = apply_sortingview_curation(sorting, uri_or_json=json_file, verbose=True)
    # print("Curated:", sorting_curated_json.get_unit_ids())

    # Assertions
    assert sorting_curated_json.get_unit_property(unit_id=1, key="accept")
    assert not sorting_curated_json.get_unit_property(unit_id=10, key="accept")
    assert 21 in sorting_curated_json.unit_ids


def test_label_inheritance_int():
    """
    Test curation for label inheritance for integer unit IDs.
    """
    # Setup
    sampling_frequency = 30000.0
    duration = 20.0
    num_timepoints = int(sampling_frequency * duration)
    num_spikes = 1000
    num_units = 7
    times = np.int_(np.sort(np.random.uniform(0, num_timepoints, num_spikes)))
    labels = np.random.randint(1, 1 + num_units, size=num_spikes)  # 7 units: 1 to 7

    sorting = se.NumpySorting.from_times_labels(times, labels, sampling_frequency)

    json_file = parent_folder / "sv-sorting-curation-int.json"
    sorting_merge = apply_sortingview_curation(sorting, uri_or_json=json_file)

    # Assertions for merged units
    # print(f"Merge only: {sorting_merge.get_unit_ids()}")
    assert sorting_merge.get_unit_property(unit_id=8, key="mua")  # 8 = merged unit of 1 and 2
    assert not sorting_merge.get_unit_property(unit_id=8, key="reject")
    assert not sorting_merge.get_unit_property(unit_id=8, key="noise")
    assert not sorting_merge.get_unit_property(unit_id=8, key="accept")

    assert not sorting_merge.get_unit_property(unit_id=9, key="mua")  # 9 = merged unit of 3 and 4
    assert sorting_merge.get_unit_property(unit_id=9, key="reject")
    assert sorting_merge.get_unit_property(unit_id=9, key="noise")
    assert not sorting_merge.get_unit_property(unit_id=9, key="accept")

    assert not sorting_merge.get_unit_property(unit_id=10, key="mua")  # 10 = merged unit of 5 and 6
    assert not sorting_merge.get_unit_property(unit_id=10, key="reject")
    assert not sorting_merge.get_unit_property(unit_id=10, key="noise")
    assert sorting_merge.get_unit_property(unit_id=10, key="accept")

    # Assertions for exclude_labels
    sorting_exclude_noise = apply_sortingview_curation(sorting, uri_or_json=json_file, exclude_labels=["noise"])
    # print(f"Exclude noise: {sorting_exclude_noise.get_unit_ids()}")
    assert 9 not in sorting_exclude_noise.get_unit_ids()

    # Assertions for include_labels
    sorting_include_accept = apply_sortingview_curation(sorting, uri_or_json=json_file, include_labels=["accept"])
    # print(f"Include accept: {sorting_include_accept.get_unit_ids()}")
    assert 8 not in sorting_include_accept.get_unit_ids()
    assert 9 not in sorting_include_accept.get_unit_ids()
    assert 10 in sorting_include_accept.get_unit_ids()


def test_label_inheritance_str():
    """
    Test curation for label inheritance for string unit IDs.
    """
    sampling_frequency = 30000.0
    duration = 20.0
    num_timepoints = int(sampling_frequency * duration)
    num_spikes = 1000
    times = np.int_(np.sort(np.random.uniform(0, num_timepoints, num_spikes)))
    labels = np.random.choice(["a", "b", "c", "d", "e", "f", "g"], size=num_spikes)

    sorting = se.NumpySorting.from_times_labels(times, labels, sampling_frequency)
    # print(f"Sorting: {sorting.get_unit_ids()}")

    # Apply curation
    json_file = parent_folder / "sv-sorting-curation-str.json"
    sorting_merge = apply_sortingview_curation(sorting, uri_or_json=json_file, verbose=True)

    # Assertions for merged units
    # print(f"Merge only: {sorting_merge.get_unit_ids()}")
    assert sorting_merge.get_unit_property(unit_id="a-b", key="mua")
    assert not sorting_merge.get_unit_property(unit_id="a-b", key="reject")
    assert not sorting_merge.get_unit_property(unit_id="a-b", key="noise")
    assert not sorting_merge.get_unit_property(unit_id="a-b", key="accept")

    assert not sorting_merge.get_unit_property(unit_id="c-d", key="mua")
    assert sorting_merge.get_unit_property(unit_id="c-d", key="reject")
    assert sorting_merge.get_unit_property(unit_id="c-d", key="noise")
    assert not sorting_merge.get_unit_property(unit_id="c-d", key="accept")

    assert not sorting_merge.get_unit_property(unit_id="e-f", key="mua")
    assert not sorting_merge.get_unit_property(unit_id="e-f", key="reject")
    assert not sorting_merge.get_unit_property(unit_id="e-f", key="noise")
    assert sorting_merge.get_unit_property(unit_id="e-f", key="accept")

    # Assertions for exclude_labels
    sorting_exclude_noise = apply_sortingview_curation(sorting, uri_or_json=json_file, exclude_labels=["noise"])
    # print(f"Exclude noise: {sorting_exclude_noise.get_unit_ids()}")
    assert "c-d" not in sorting_exclude_noise.get_unit_ids()

    # Assertions for include_labels
    sorting_include_accept = apply_sortingview_curation(sorting, uri_or_json=json_file, include_labels=["accept"])
    # print(f"Include accept: {sorting_include_accept.get_unit_ids()}")
    assert "a-b" not in sorting_include_accept.get_unit_ids()
    assert "c-d" not in sorting_include_accept.get_unit_ids()
    assert "e-f" in sorting_include_accept.get_unit_ids()


if __name__ == "__main__":
    # generate_sortingview_curation_dataset()
    test_sha1_curation()
    test_gh_curation()
    test_json_curation()
    test_false_positive_curation()
    test_label_inheritance_int()
    test_label_inheritance_str()
