"""
Some simple function to retrieve public datasets with datalad
"""

import warnings
from .globals import get_global_dataset_folder, is_set_global_dataset_folder


def download_dataset(repo=None, remote_path=None, local_folder=None, update_if_exists=False, unlock=False):
    """
    Function to download dataset from a remote repository using datalad.

    Parameters
    ----------
    repo : str, optional
        The repository to download the dataset from,
        defaults to: 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path : str
        A specific subdirectory in the repository to download (e.g. Mearec, SpikeGLX, etc)
        If not provided, the function returns None
    local_folder : str, optional
        The destination folder / directory to download the dataset to.
        defaults to the path "get_global_dataset_folder()" / f{repo_name} (see `spikeinterface.core.globals`) 
    update_if_exists : bool, optional
        Forces re-download of the dataset if it already exists, by default False
    unlock : bool, optional
        Use to enable the edition of the downloaded file content, by default False

    Returns
    -------
    str
        The local path to the downloaded dataset
    """
    import datalad.api
    from datalad.support.gitrepo import GitRepo

    if repo is None:
        #  print('Use gin NeuralEnsemble/ephy_testing_data')
        repo = "https://gin.g-node.org/NeuralEnsemble/ephy_testing_data"

    if local_folder is None:
        base_local_folder = get_global_dataset_folder()
        base_local_folder.mkdir(exist_ok=True)
        local_folder = base_local_folder / repo.split("/")[-1]

    if local_folder.exists() and GitRepo.is_valid_repo(local_folder):
        dataset = datalad.api.Dataset(path=local_folder)
        # make sure git repo is in clean state
        repo = dataset.repo
        if update_if_exists:
            repo.call_git(["checkout", "--force", "master"])
            dataset.update(merge=True)
    else:
        dataset = datalad.api.install(path=local_folder, source=repo)

    if remote_path is None:
        warnings.warn(message="No remote path provided, returning None")
        return

    local_path = local_folder / remote_path

    # This downloads the data set content
    dataset.get(remote_path)

    # Unlock files of a dataset in order to be able to edit the actual content
    if unlock:
        dataset.unlock(remote_path, recursive=True)

    return local_path
