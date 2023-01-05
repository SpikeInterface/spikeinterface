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


ON_GITHUB = bool(os.getenv('GITHUB_ACTIONS'))
KACHERY_CLOUD_SET = bool(os.getenv('KACHERY_CLOUD_CLIENT_ID')) and bool(os.getenv('KACHERY_CLOUD_PRIVATE_KEY'))


set_global_tmp_folder(cache_folder)

# this needs to be run only once
def generate_sortingview_curation_dataset():
    import spikeinterface.widgets as sw

    local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
    recording, sorting = read_mearec(local_path)

    we = si.extract_waveforms(recording, sorting, folder=cache_folder / "waveforms")

    _ = compute_spike_amplitudes(we)
    _ = compute_correlograms(we)
    _ = compute_template_similarity(we)
    _ = compute_unit_locations(we)

    # plot_sorting_summary with curation
    w = sw.plot_sorting_summary(we, curation=True, backend="sortingview")


@pytest.mark.skipif(ON_GITHUB and not KACHERY_CLOUD_SET, reason="Kachery cloud secrets not available")
def test_sortingview_curation():
    local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
    _, sorting = read_mearec(local_path)

    jot_uri = "jot://cENcosqWTArO"
    # curation link: 
    # https://figurl.org/f?v=gs://figurl/spikesortingview-10&d=sha1://1ba03f81e62ec7cb2e3e46898830f92cdf5e026f&label=SpikeInterface%20-%20Sorting%20Summary&s={%22sortingCuration%22:%22jot://cENcosqWTArO%22}
    sorting_curated_jot = apply_sortingview_curation(sorting, uri_or_json=jot_uri, verbose=True)

    assert len(sorting_curated_jot.unit_ids) == 7
    assert "#0-#1" in sorting_curated_jot.unit_ids
    assert "#5-#6-#7" in sorting_curated_jot.unit_ids
    assert "accept" in sorting_curated_jot.get_property_keys()
    assert "noise" in sorting_curated_jot.get_property_keys()
    assert "reject" in sorting_curated_jot.get_property_keys()

    sorting_curated_jot_accepted = apply_sortingview_curation(sorting, uri_or_json=jot_uri, include_labels=["accept"])
    sorting_curated_jot_rejected = apply_sortingview_curation(sorting, uri_or_json=jot_uri, exclude_labels=["reject"])
    sorting_curated_jot_rejected1 = apply_sortingview_curation(sorting, uri_or_json=jot_uri,
                                                               exclude_labels=["noise", "reject"])
    assert len(sorting_curated_jot_accepted.unit_ids) == 4
    assert len(sorting_curated_jot_rejected.unit_ids) == 5
    assert len(sorting_curated_jot_rejected1.unit_ids) == 4

    # curation_link: 
    # https://figurl.org/f?v=gs://figurl/spikesortingview-10&d=sha1://1ba03f81e62ec7cb2e3e46898830f92cdf5e026f&label=SpikeInterface%20-%20Sorting%20Summary&s={%22sortingCuration%22:%22sha1://59feb326204cf61356f1a2eb31f04d8e0177c4f1%22}
    sha_uri = "sha1://59feb326204cf61356f1a2eb31f04d8e0177c4f1"
    sorting_curated_sha = apply_sortingview_curation(sorting, uri_or_json=sha_uri, verbose=True)

    assert len(sorting_curated_sha.unit_ids) == 9
    assert "#8-#9" in sorting_curated_sha.unit_ids
    assert "accept" in sorting_curated_sha.get_property_keys()
    assert "mua" in sorting_curated_sha.get_property_keys()
    assert "artifact" in sorting_curated_sha.get_property_keys()

    sorting_curated_sha_accepted = apply_sortingview_curation(sorting, uri_or_json=sha_uri, include_labels=["accept"])
    sorting_curated_sha_mua = apply_sortingview_curation(sorting, uri_or_json=sha_uri, exclude_labels=["mua"])
    sorting_curated_sha_mua1 = apply_sortingview_curation(sorting, uri_or_json=sha_uri,
                                                          exclude_labels=["artifact", "mua"])
    assert len(sorting_curated_sha_accepted.unit_ids) == 3
    assert len(sorting_curated_sha_mua.unit_ids) == 6
    assert len(sorting_curated_sha_mua1.unit_ids) == 5


if __name__ == "__main__":
    # generate_sortingview_curation_dataset()
    test_sortingview_curation()
