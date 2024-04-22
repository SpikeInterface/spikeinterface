"""
Some simple function to retrieve public datasets with datalad
"""

from __future__ import annotations

from pathlib import Path

from .globals import get_global_dataset_folder


def download_dataset(
    repo: str = "https://gin.g-node.org/NeuralEnsemble/ephy_testing_data",
    remote_path: str = "mearec/mearec_test_10s.h5",
    local_folder: Path | None = None,
    update_if_exists: bool = False,
    unlock: bool = False,
) -> Path:
    """
    Function to download dataset from a remote repository using datalad.

    Parameters
    ----------
    repo : str, default: "https://gin.g-node.org/NeuralEnsemble/ephy_testing_data"
        The repository to download the dataset from
    remote_path : str, default: "mearec/mearec_test_10s.h5"
        A specific subdirectory in the repository to download (e.g. Mearec, SpikeGLX, etc)
    local_folder : str, default: None
        The destination folder / directory to download the dataset to.
        defaults to the path "get_global_dataset_folder()" / f{repo_name} (see `spikeinterface.core.globals`)
    update_if_exists : bool, default: False
        Forces re-download of the dataset if it already exists, default: False
    unlock : bool, default: False
        Use to enable the edition of the downloaded file content, default: False

    Returns
    -------
    Path
        The local path to the downloaded dataset
    """
    import datalad.api
    from datalad.support.gitrepo import GitRepo

    if local_folder is None:
        base_local_folder = get_global_dataset_folder()
        base_local_folder.mkdir(exist_ok=True, parents=True)
        local_folder = base_local_folder / repo.split("/")[-1]

    local_folder = Path(local_folder)
    if local_folder.exists() and GitRepo.is_valid_repo(local_folder):
        dataset = datalad.api.Dataset(path=local_folder)
        # make sure git repo is in clean state
        repo = dataset.repo
        if update_if_exists:
            repo.call_git(["checkout", "--force", "master"])
            dataset.update(merge=True)
    else:
        dataset = datalad.api.install(path=local_folder, source=repo)

    local_path = local_folder / remote_path

    # This downloads the data set content
    dataset.get(remote_path)

    # Unlock files of a dataset in order to be able to edit the actual content
    if unlock:
        dataset.unlock(remote_path, recursive=True)

    return local_path
