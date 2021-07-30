"""
Some simple function to retrieve public datasets.
"""

from .default_folders import get_global_dataset_folder, is_set_global_dataset_folder

try:
    import datalad.api

    HAVE_DATALAD = True
except:
    HAVE_DATALAD = False


def download_dataset(repo=None, remote_path=None, local_folder=None, update_if_exists=False):
    if repo is None:
        #  print('Use gin NeuralEnsemble/ephy_testing_data')
        repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'

    if local_folder is None:
        base_local_folder = get_global_dataset_folder()
        base_local_folder.mkdir(exist_ok=True)
        #  if not is_set_global_dataset_folder():
        #  print(f'Local folder is {base_local_folder}, Use set_global_dataset_folder() to set it globaly')
        local_folder = base_local_folder / repo.split('/')[-1]

    if local_folder.exists():
        dataset = datalad.api.Dataset(path=local_folder)
        if update_if_exists:
            dataset.update(merge=True)
    else:
        dataset = datalad.api.install(path=local_folder,
                                      source='https://gin.g-node.org/NeuralEnsemble/ephy_testing_data')

    if remote_path is None:
        print('Bad boy: you have to provide "remote_path"')
        return

    dataset.get(remote_path)

    local_path = local_folder / remote_path

    return local_path
