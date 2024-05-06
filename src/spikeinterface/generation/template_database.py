import zarr
import numpy as np
import pandas as pd

from spikeinterface.core.template import Templates


def fetch_template_dataset(dataset="test_templates") -> Templates:
    """
    Fetch a template dataset from the spikeinterface template database.
    A dataset is a collection of templates with associated metadata for one specific recording.

    Parameters
    ----------
    dataset : str, default: "test_templates"
        The name of the dataset to fetch.
        The dataset must be available in the spikeinterface template database.

    Returns
    -------
    Templates
        _description_
    """

    s3_path = f"s3://spikeinterface-template-database/{dataset}/"
    zarr_group = zarr.open_consolidated(s3_path, storage_options={"anon": True})

    templates_object = Templates.from_zarr_group(zarr_group)

    return templates_object


def fetch_templates_info() -> pd.DataFrame:
    """
    Fetch the information about the templates in the spikeinterface template database.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the template information.
    """
    s3_path = "s3://spikeinterface-template-database/templates.csv"
    df = pd.read_csv(s3_path, storage_options={"anon": True})

    return df


def list_avaliabe_datasets() -> list:
    """
    List all available datasets in the spikeinterface template database.

    Returns
    -------
    list
        List of available datasets.
    """
    df = fetch_templates_info()
    datasets = np.unique(df["dataset"]).tolist()

    return datasets


def get_templates_from_database(template_df_or_indices: pd.DataFrame | list[int] | np.ndarray) -> Templates:
    """
    Retrieve templates from the spikeinterface template database.

    Parameters
    ----------
    template_info : pd.DataFrame or array-like
        Dataframe containing the template information, obtained by slicing/querying the output of fetch_templates_info.

    Returns
    -------
    Templates
        The templates object.
    """
    templates_array = []
    if isinstance(template_df_or_indices, pd.DataFrame):
        template_info = template_df_or_indices
    else:
        template_info = fetch_templates_info().iloc[template_df_or_indices]
    requested_datasets = np.unique(template_info["dataset"]).tolist()
    print(f"Fetching templates from {len(requested_datasets)} datasets")

    nbefore = None
    sampling_frequency = None
    channel_locations = None
    probe = None
    channel_ids = None

    for dataset in requested_datasets:
        templates = fetch_template_dataset(dataset)

        # check consisency across datasets
        if nbefore is None:
            nbefore = templates.nbefore
        if channel_locations is None:
            channel_locations = templates.get_channel_locations()
        if sampling_frequency is None:
            sampling_frequency = templates.sampling_frequency
        if probe is None:
            probe = templates.probe
        if channel_ids is None:
            channel_ids = templates.channel_ids
        current_nbefore = templates.nbefore
        current_channel_locations = templates.get_channel_locations()
        current_sampling_frequency = templates.sampling_frequency

        assert (
            current_nbefore == nbefore
        ), f"Number of samples before the peak is not consistent across datasets: {current_nbefore} != {nbefore}"
        assert (
            current_sampling_frequency == sampling_frequency
        ), f"Sampling frequency is not consistent across datasets: {current_sampling_frequency} != {sampling_frequency}"
        assert np.array_equal(
            current_channel_locations - current_channel_locations[0],
            channel_locations - channel_locations[0],
        ), "Channel locations are not consistent across datasets"

        template_indices = template_info[template_info["dataset"] == dataset]["template_index"]
        templates_array.append(templates.templates_array[template_indices, :, :])
    templates_array = np.concatenate(templates_array, axis=0)
    templates = Templates(
        templates_array,
        sampling_frequency=sampling_frequency,
        channel_ids=channel_ids,
        nbefore=nbefore,
        probe=probe,
    )

    return templates
