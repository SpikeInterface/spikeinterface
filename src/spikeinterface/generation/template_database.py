from spikeinterface.core.template import Templates
import zarr


def fetch_templates_from_database(dataset="test_templates") -> Templates:

    s3_path = f"s3://spikeinterface-template-database/{dataset}/"
    zarr_group = zarr.open_consolidated(s3_path, storage_options={"anon": True})

    templates_object = Templates.from_zarr_group(zarr_group)

    return templates_object
