import pytest
from spikeinterface.core import download_dataset
import importlib.util


@pytest.mark.skipif(
    importlib.util.find_spec("pooch") is None or importlib.util.find_spec("datalad") is None,
    reason="Either pooch or datalad is not installed",
)
def test_download_dataset():
    repo = "https://gin.g-node.org/NeuralEnsemble/ephy_testing_data"
    remote_path = "mearec"

    # local_folder automatic
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)


if __name__ == "__main__":
    test_download_dataset()
