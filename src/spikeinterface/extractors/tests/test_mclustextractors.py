import importlib.util
import shutil

import pytest

from spikeinterface.core import download_dataset
from spikeinterface.extractors.extractor_classes import read_mclust


@pytest.mark.skipif(
    importlib.util.find_spec("pooch") is None or importlib.util.find_spec("datalad") is None,
    reason="Either pooch or datalad is not installed",
)
def test_read_mclust(tmp_path):
    # The mclust dataset on gin stores big-endian uint64 timestamps in the plain `.t` suffix
    # (MClust 3.x). This previously made read_mclust return zero units: the extension was always
    # detected as `t64` and the uint64 data was misread as uint32 (GH-4602).
    folder = download_dataset(remote_path="mclust")

    # The dataset has two `.t` files whose names both end in `_1`, so they parse to the same unit
    # id. Isolate one of them (the smaller TT06 file) so read_mclust sees a single unit.
    single_folder = tmp_path / "mclust_single"
    single_folder.mkdir()
    shutil.copy(folder / "M040-2020-04-28-TT06_1.t", single_folder / "M040-2020-04-28-TT06_1.t")

    sorting = read_mclust(single_folder, sampling_frequency=32000.0)

    assert sorting.get_num_units() == 1
    assert len(sorting.get_unit_spike_train(unit_id=1)) == 5988
