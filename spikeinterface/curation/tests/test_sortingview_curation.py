import pytest
from pathlib import Path
import os

import spikeinterface as si
from spikeinterface.extractors import read_mearec
from spikeinterface import set_global_tmp_folder
from spikeinterface.postprocessing import (compute_correlograms, compute_unit_locations,
                                           compute_template_similarity, compute_spike_amplitudes)
from spikeinterface.curation import apply_sortingview_curation

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "curation"
else:
    cache_folder = Path("cache_folder") / "curation"

parent_folder = Path(__file__).parent

ON_GITHUB = bool(os.getenv('GITHUB_ACTIONS'))
KACHERY_CLOUD_SET = bool(os.getenv('KACHERY_CLOUD_CLIENT_ID')) and bool(os.getenv('KACHERY_CLOUD_PRIVATE_KEY'))


set_global_tmp_folder(cache_folder)

# this needs to be run only once
def generate_sortingview_curation_dataset():
    import spikeinterface.widgets as sw

    local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
    recording, sorting = read_mearec(local_path)

    we = si.extract_waveforms(recording, sorting, folder=None, mode="memory")

    _ = compute_spike_amplitudes(we)
    _ = compute_correlograms(we)
    _ = compute_template_similarity(we)
    _ = compute_unit_locations(we)

    # plot_sorting_summary with curation
    w = sw.plot_sorting_summary(we, curation=True, backend="sortingview")

    # curation_link:
    # https://figurl.org/f?v=gs://figurl/spikesortingview-10&d=sha1://bd53f6b707f8121cadc901562a89b67aec81cc81&label=SpikeInterface%20-%20Sorting%20Summary


@pytest.mark.skipif(ON_GITHUB and not KACHERY_CLOUD_SET, reason="Kachery cloud secrets not available")
def test_gh_curation():
    local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
    _, sorting = read_mearec(local_path)

    # from GH
    # curated link:
    # https://figurl.org/f?v=gs://figurl/spikesortingview-10&d=sha1://bd53f6b707f8121cadc901562a89b67aec81cc81&label=SpikeInterface%20-%20Sorting%20Summary&s={%22sortingCuration%22:%22gh://alejoe91/spikeinterface/fix-codecov/spikeinterface/curation/tests/sv-sorting-curation.json%22}
    gh_uri = "gh://SpikeInterface/spikeinterface/master/spikeinterface/curation/tests/sv-sorting-curation.json"
    sorting_curated_gh = apply_sortingview_curation(sorting, uri_or_json=gh_uri, verbose=True)
    print(f"From GH: {sorting_curated_gh}")

    assert len(sorting_curated_gh.unit_ids) == 9
    assert "#8-#9" in sorting_curated_gh.unit_ids
    assert "accept" in sorting_curated_gh.get_property_keys()
    assert "mua" in sorting_curated_gh.get_property_keys()
    assert "artifact" in sorting_curated_gh.get_property_keys()

    sorting_curated_gh_accepted = apply_sortingview_curation(sorting, uri_or_json=gh_uri, include_labels=["accept"])
    sorting_curated_gh_mua = apply_sortingview_curation(sorting, uri_or_json=gh_uri, exclude_labels=["mua"])
    sorting_curated_gh_art_mua = apply_sortingview_curation(sorting, uri_or_json=gh_uri,
                                                            exclude_labels=["artifact", "mua"])
    assert len(sorting_curated_gh_accepted.unit_ids) == 3
    assert len(sorting_curated_gh_mua.unit_ids) == 6
    assert len(sorting_curated_gh_art_mua.unit_ids) == 5


@pytest.mark.skipif(ON_GITHUB and not KACHERY_CLOUD_SET, reason="Kachery cloud secrets not available")
def test_sha1_curation():
    local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
    _, sorting = read_mearec(local_path)

    # from SHA1
    # curated link:
    # https://figurl.org/f?v=gs://figurl/spikesortingview-10&d=sha1://bd53f6b707f8121cadc901562a89b67aec81cc81&label=SpikeInterface%20-%20Sorting%20Summary&s={%22sortingCuration%22:%22sha1://1182ba19671fcc7d3f8e0501b0f8c07fb9736c22%22}
    sha1_uri = "sha1://1182ba19671fcc7d3f8e0501b0f8c07fb9736c22"
    sorting_curated_sha1 = apply_sortingview_curation(sorting, uri_or_json=sha1_uri, verbose=True)
    print(f"From SHA: {sorting_curated_sha1}")

    assert len(sorting_curated_sha1.unit_ids) == 9
    assert "#8-#9" in sorting_curated_sha1.unit_ids
    assert "accept" in sorting_curated_sha1.get_property_keys()
    assert "mua" in sorting_curated_sha1.get_property_keys()
    assert "artifact" in sorting_curated_sha1.get_property_keys()

    sorting_curated_sha1_accepted = apply_sortingview_curation(sorting, uri_or_json=sha1_uri, include_labels=["accept"])
    sorting_curated_sha1_mua = apply_sortingview_curation(sorting, uri_or_json=sha1_uri, exclude_labels=["mua"])
    sorting_curated_sha1_art_mua = apply_sortingview_curation(sorting, uri_or_json=sha1_uri,
                                                              exclude_labels=["artifact", "mua"])
    assert len(sorting_curated_sha1_accepted.unit_ids) == 3
    assert len(sorting_curated_sha1_mua.unit_ids) == 6
    assert len(sorting_curated_sha1_art_mua.unit_ids) == 5


def test_json_curation():
    local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
    _, sorting = read_mearec(local_path)

    # from curation.json
    json_file = parent_folder / "sv-sorting-curation.json"
    sorting_curated_json = apply_sortingview_curation(sorting, uri_or_json=json_file, verbose=True)
    print(f"From JSON: {sorting_curated_json}")

    assert len(sorting_curated_json.unit_ids) == 9
    assert "#8-#9" in sorting_curated_json.unit_ids
    assert "accept" in sorting_curated_json.get_property_keys()
    assert "mua" in sorting_curated_json.get_property_keys()
    assert "artifact" in sorting_curated_json.get_property_keys()

    sorting_curated_json_accepted = apply_sortingview_curation(sorting, uri_or_json=json_file, include_labels=["accept"])
    sorting_curated_json_mua = apply_sortingview_curation(sorting, uri_or_json=json_file, exclude_labels=["mua"])
    sorting_curated_json_mua1 = apply_sortingview_curation(sorting, uri_or_json=json_file,
                                                           exclude_labels=["artifact", "mua"])
    assert len(sorting_curated_json_accepted.unit_ids) == 3
    assert len(sorting_curated_json_mua.unit_ids) == 6
    assert len(sorting_curated_json_mua1.unit_ids) == 5


if __name__ == "__main__":
    # generate_sortingview_curation_dataset()
    test_sha1_curation()
    test_gh_curation()
    test_json_curation()
