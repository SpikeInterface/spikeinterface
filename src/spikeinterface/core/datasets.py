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
) -> Path:
    """
    Function to download dataset from a remote repository using a combination of datalad and pooch.

    Pooch is designed to download single files from a remote repository.
    Because our datasets in gin sometimes point just to a folder, we still use datalad to download
    a list of all the files in the folder and then use pooch to download them one by one.

    Parameters
    ----------
    repo : str, default: "https://gin.g-node.org/NeuralEnsemble/ephy_testing_data"
        The repository to download the dataset from
    remote_path : str, default: "mearec/mearec_test_10s.h5"
        A specific subdirectory in the repository to download (e.g. Mearec, SpikeGLX, etc)
    local_folder : str, optional
        The destination folder / directory to download the dataset to.
        if None, then the path "get_global_dataset_folder()" / f{repo_name} is used (see `spikeinterface.core.globals`)
    update_if_exists : bool, default: False
        Forces re-download of the dataset if it already exists, default: False

    Returns
    -------
    Path
        The local path to the downloaded dataset

    Notes
    -----
    The reason we use pooch is because have had problems with datalad not being able to download
    data on windows machines. Specially in the CI.

    See https://handbook.datalad.org/en/latest/intro/windows.html
    """
    import pooch
    import datalad.api
    from datalad.support.gitrepo import GitRepo

    if local_folder is None:
        base_local_folder = get_global_dataset_folder()
        base_local_folder.mkdir(exist_ok=True, parents=True)
        local_folder = base_local_folder / repo.split("/")[-1]
        local_folder.mkdir(exist_ok=True, parents=True)
    else:
        if not local_folder.is_dir():
            local_folder.mkdir(exist_ok=True, parents=True)

    local_folder = Path(local_folder)
    if local_folder.exists() and GitRepo.is_valid_repo(local_folder):
        dataset = datalad.api.Dataset(path=local_folder)
        # make sure git repo is in clean state
        repo_object = dataset.repo
        if update_if_exists:
            repo_object.call_git(["checkout", "--force", "master"])
            dataset.update(merge=True)
    else:
        dataset = datalad.api.install(path=local_folder, source=repo)

    local_path = local_folder / remote_path

    if local_path.is_dir():
        files_to_download = []
        files_to_download = [file_path for file_path in local_path.rglob("*") if not file_path.is_dir()]
    else:
        files_to_download = [local_path]

    for file_path in files_to_download:
        remote_path = file_path.relative_to(local_folder)
        url = f"{repo}/raw/master/{remote_path}"
        file_path = local_folder / remote_path
        file_path.unlink(missing_ok=True)

        full_path = pooch.retrieve(
            url=url,
            fname=str(file_path),
            path=local_folder,
            known_hash=None,
            progressbar=True,
        )

        assert Path(full_path).is_file(), f"File {full_path} not found"

    return local_path
