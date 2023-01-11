"""
Some simple function to retrieve public datasets.
"""

from .globals import get_global_dataset_folder, is_set_global_dataset_folder


def download_dataset(repo=None, remote_path=None, local_folder=None, update_if_exists=False,
                     unlock=False):
    import datalad.api
    from datalad.support.gitrepo import GitRepo


    if repo is None:
        #  print('Use gin NeuralEnsemble/ephy_testing_data')
        repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'

    if local_folder is None:
        base_local_folder = get_global_dataset_folder()
        base_local_folder.mkdir(exist_ok=True)
        #  if not is_set_global_dataset_folder():
        #  print(f'Local folder is {base_local_folder}, Use set_global_dataset_folder() to set it globally')
        local_folder = base_local_folder / repo.split('/')[-1]

    if local_folder.exists() and GitRepo.is_valid_repo(local_folder):
        dataset = datalad.api.Dataset(path=local_folder)
        # make sure git repo is in clean state
        repo = dataset.repo
        if update_if_exists:
            repo.call_git(['checkout', '--force', 'master'])
            dataset.update(merge=True)
    else:
        dataset = datalad.api.install(path=local_folder,
                                      source=repo)

    if remote_path is None:
        print('Bad boy: you have to provide "remote_path"')
        return

    local_path = local_folder / remote_path

    dataset.get(remote_path)

    # unlocking is necessary for binding volume to containers
    if unlock:
        dataset.unlock(remote_path, recursive=True)

    return local_path
