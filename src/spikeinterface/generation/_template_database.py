from spikeinterface.core.template import Templates
from spikeinterface.core import generate_sorting, InjectTemplatesRecording
import zarr


def fetch_templates_from_database(dataset="test_templates"):

    import s3fs

    s3 = s3fs.S3FileSystem(anon=False, client_kwargs={"region_name": "us-east-2"})

    # Specify the S3 bucket and path where your Zarr dataset is stored
    store = s3fs.S3Map(root=f"spikeinterface-template-database/{dataset}", s3=s3)

    # Load the Zarr group from S3
    zarr_group = zarr.open(store, mode="r")

    templates_object = Templates.from_zarr_group(zarr_group)

    return templates_object


def generate_recording_from_template_database(selected_unit_inidces=None, dataset="test_templates", durations=None):

    durations = durations or [10.0]

    templates_object = fetch_templates_from_database(dataset=dataset)

    if selected_unit_inidces:
        selected_templates = templates_object.templates_array[selected_unit_inidces, :, :]
    else:
        selected_templates = templates_object.templates_array

    num_units = selected_templates.shape[0]
    sampling_frequency = templates_object.sampling_frequency
    sorting = generate_sorting(num_units=num_units, sampling_frequency=sampling_frequency, durations=durations)

    nbefore = templates_object.nbefore
    num_samples = durations[0] * sampling_frequency
    recording = InjectTemplatesRecording(
        sorting=sorting,
        templates=selected_templates,
        nbefore=nbefore,
        num_samples=[num_samples],
    )

    return recording
