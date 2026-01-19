from __future__ import annotations

from typing import Optional
from pathlib import Path
import warnings

import numpy as np

from spikeinterface.core import (
    BaseSorting,
    BaseSortingSegment,
    read_python,
    generate_ground_truth_recording,
    ChannelSparsity,
    ComputeTemplates,
    create_sorting_analyzer,
    SortingAnalyzer,
)
from spikeinterface.core.core_tools import define_function_from_class

from spikeinterface.postprocessing import ComputeSpikeAmplitudes, ComputeSpikeLocations
from probeinterface import read_prb, Probe


class BasePhyKilosortSortingExtractor(BaseSorting):
    """Base SortingExtractor for Phy and Kilosort output folder.

    Parameters
    ----------
    folder_path : str or Path
        Path to the output Phy/Kilosort folder (containing the params.py)
    exclude_cluster_groups : list or str, default: None
        Cluster groups to exclude (e.g. "noise" or ["noise", "mua"]).
    keep_good_only : bool, default: True
        Whether to only keep good units.
    remove_empty_units : bool, default: False
        If True, empty units are removed from the sorting extractor.
    load_all_cluster_properties : bool, default: True
        If True, all cluster properties are loaded from the tsv/csv files.

    Notes
    -----
    This extractor loads cluster properties from CSV/TSV files to enrich the sorting
    extractor with unit metadata such as quality labels, groups, and Kilosort metrics.

    Cluster information is loaded in the following priority order:
    1. From a dedicated cluster_info.csv/.tsv file if present
    2. From all .csv/.tsv files in the folder that contain a 'cluster_id' column
       Typical files include cluster_group.tsv, cluster_info.tsv, cluster_KSLabel.tsv
       Files without cluster_id column are automatically skipped
    3. If no files are found, minimal cluster info is generated with 'unsorted' labels

    The cluster_id column is used as the merge key to combine properties from multiple files.
    All loaded properties are added to the sorting extractor as unit properties, with some
    renamed for SpikeInterface conventions: 'group' becomes 'quality', 'cluster_id'
    becomes 'original_cluster_id'. These properties can be accessed via ``sorting.get_property()``
    function.
    """

    installation_mesg = (
        "To use the PhySortingExtractor install pandas: \n\n pip install pandas\n\n"  # error message when not installed
    )

    def __init__(
        self,
        folder_path: Path | str,
        exclude_cluster_groups: Optional[list[str] | str] = None,
        keep_good_only: bool = False,
        remove_empty_units: bool = False,
        load_all_cluster_properties: bool = True,
    ):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(self.installation_mesg)

        phy_folder = Path(folder_path)
        spike_times = np.load(phy_folder / "spike_times.npy").astype(int)

        if (phy_folder / "spike_clusters.npy").is_file():
            spike_clusters = np.load(phy_folder / "spike_clusters.npy")
        else:
            spike_clusters = np.load(phy_folder / "spike_templates.npy")

        # spike_times and spike_clusters can be 2d sometimes --> convert to 1d.
        spike_times = np.atleast_1d(spike_times.squeeze())
        spike_clusters = np.atleast_1d(spike_clusters.squeeze())

        clust_id = np.unique(spike_clusters)
        unique_unit_ids = [int(c) for c in clust_id]
        params = read_python(str(phy_folder / "params.py"))
        sampling_frequency = params["sample_rate"]

        # try to load cluster info
        cluster_info_files = [
            p for p in phy_folder.iterdir() if p.suffix in [".csv", ".tsv"] and "cluster_info" in p.name
        ]

        if len(cluster_info_files) == 1:
            # load properties from cluster_info file
            cluster_info_file = cluster_info_files[0]
            if cluster_info_file.suffix == ".tsv":
                delimiter = "\t"
            else:
                delimiter = ","
            cluster_info = pd.read_csv(cluster_info_file, delimiter=delimiter)
        else:
            # load properties from other tsv/csv files
            all_property_files = [p for p in phy_folder.iterdir() if p.suffix in [".csv", ".tsv"]]

            cluster_info = None
            for file in all_property_files:
                if file.suffix == ".tsv":
                    delimiter = "\t"
                else:
                    delimiter = ","
                new_property = pd.read_csv(file, delimiter=delimiter)

                # Only merge files that contain a cluster_id column
                # This prevents KeyError when extraneous files don't have cluster_id
                # Typical aggregated files include cluster_group.tsv, cluster_info.tsv, cluster_KSLabel.tsv
                # See Phy docs: https://phy.readthedocs.io/en/latest/sorting_user_guide/
                # See: https://github.com/SpikeInterface/spikeinterface/issues/4124
                if "cluster_id" not in new_property.columns:
                    continue

                if cluster_info is None:
                    cluster_info = new_property
                else:
                    cluster_info = pd.merge(cluster_info, new_property, on="cluster_id", suffixes=[None, "_repeat"])

        # in case no tsv/csv files are found populate cluster info with minimal info
        if cluster_info is None:
            cluster_info = pd.DataFrame({"cluster_id": unique_unit_ids})
            cluster_info["group"] = ["unsorted"] * len(unique_unit_ids)

        if exclude_cluster_groups is not None:
            if isinstance(exclude_cluster_groups, str):
                cluster_info = cluster_info.query(f"group != '{exclude_cluster_groups}'")
            elif isinstance(exclude_cluster_groups, list):
                if len(exclude_cluster_groups) > 0:
                    for exclude_group in exclude_cluster_groups:
                        cluster_info = cluster_info.query(f"group != '{exclude_group}'")

        if keep_good_only and "KSLabel" in cluster_info.columns:
            cluster_info = cluster_info.query("KSLabel == 'good'")

        if "cluster_id" not in cluster_info.columns:
            assert "id" in cluster_info.columns, "Couldn't find cluster ids in the tsv files!"
            cluster_info.loc[:, "cluster_id"] = cluster_info["id"].values
            del cluster_info["id"]

        if remove_empty_units:
            cluster_info = cluster_info.query(f"cluster_id in {unique_unit_ids}")

        # update spike clusters and times values
        bad_clusters = [clust for clust in clust_id if clust not in cluster_info["cluster_id"].values]
        spike_clusters_clean_idxs = ~np.isin(spike_clusters, bad_clusters)
        spike_clusters_clean = spike_clusters[spike_clusters_clean_idxs]
        spike_times_clean = spike_times[spike_clusters_clean_idxs]

        if "si_unit_id" in cluster_info.columns:
            unit_ids = cluster_info["si_unit_id"].values

            if np.all(np.isnan(unit_ids)):
                max_si_unit_id = -1
            else:
                max_si_unit_id = int(np.nanmax(unit_ids))

            for i, (phy_id, si_id) in enumerate(
                zip(cluster_info["cluster_id"].values, cluster_info["si_unit_id"].values)
            ):
                if np.isnan(si_id) or np.count_nonzero(cluster_info["si_unit_id"].values == si_id) != 1:
                    max_si_unit_id += 1
                    new_si_id = int(max_si_unit_id)
                else:
                    new_si_id = si_id
                unit_ids[i] = new_si_id

            # Little hack to replace values in spike_clusters_clean to spike_clusters_new very efficiently.
            from_values = cluster_info["cluster_id"].values
            sort_idx = np.argsort(from_values)
            idx = np.searchsorted(from_values, spike_clusters_clean, sorter=sort_idx)
            spike_clusters_new = unit_ids[sort_idx][idx]

            unit_ids = unit_ids.astype(int)
            spike_clusters_clean = spike_clusters_new
            del cluster_info["si_unit_id"]
        else:
            unit_ids = cluster_info["cluster_id"].values

        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        self.extra_requirements.append("pandas")

        for prop_name in cluster_info.columns:
            if prop_name in ["chan_grp", "ch_group", "channel_group"]:
                self.set_property(key="group", values=cluster_info[prop_name])
            elif prop_name == "cluster_id":
                self.set_property(key="original_cluster_id", values=cluster_info[prop_name])
            elif prop_name == "group":
                # rename group property to 'quality'
                values = cluster_info[prop_name].values.astype("str")
                self.set_property(key="quality", values=values)
            else:
                if load_all_cluster_properties:
                    # pandas loads strings with empty values as objects with NaNs
                    prop_dtype = None
                    if cluster_info[prop_name].values.dtype.kind == "O":
                        for value in cluster_info[prop_name].values:
                            if isinstance(value, (np.floating, float)) and np.isnan(
                                value
                            ):  # Blank values are encoded as 'NaN'.
                                continue

                            prop_dtype = type(value)
                            break
                        if prop_dtype is not None:
                            values_ = cluster_info[prop_name].values.astype(prop_dtype)
                        else:
                            # Could not find a valid dtype for the column. Skip it.
                            continue
                    else:
                        values_ = cluster_info[prop_name].values
                    self.set_property(key=prop_name, values=values_)

        self.annotate(phy_folder=str(phy_folder.resolve()))

        self.add_sorting_segment(PhySortingSegment(spike_times_clean, spike_clusters_clean))


class PhySortingSegment(BaseSortingSegment):
    def __init__(self, all_spikes, all_clusters):
        BaseSortingSegment.__init__(self)
        self._all_spikes = all_spikes
        self._all_clusters = all_clusters

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        start = 0 if start_frame is None else np.searchsorted(self._all_spikes, start_frame, side="left")
        end = (
            len(self._all_spikes) if end_frame is None else np.searchsorted(self._all_spikes, end_frame, side="left")
        )  # Exclude end frame

        spike_times = self._all_spikes[start:end][self._all_clusters[start:end] == unit_id]
        return np.atleast_1d(spike_times.copy().squeeze())


class PhySortingExtractor(BasePhyKilosortSortingExtractor):
    """Load Phy format data as a sorting extractor.

    Parameters
    ----------
    folder_path : str or Path
        Path to the output Phy folder (containing the params.py).
    exclude_cluster_groups : list or str, default: None
        Cluster groups to exclude (e.g. "noise" or ["noise", "mua"]).
    load_all_cluster_properties : bool, default: True
        If True, all cluster properties are loaded from the tsv/csv files.

    Returns
    -------
    extractor : PhySortingExtractor
        The loaded Sorting object.
    """

    def __init__(
        self,
        folder_path: Path | str,
        exclude_cluster_groups: Optional[list[str] | str] = None,
        load_all_cluster_properties: bool = True,
    ):
        BasePhyKilosortSortingExtractor.__init__(
            self,
            folder_path,
            exclude_cluster_groups,
            keep_good_only=False,
            load_all_cluster_properties=load_all_cluster_properties,
        )

        self._kwargs = {
            "folder_path": str(Path(folder_path).absolute()),
            "exclude_cluster_groups": exclude_cluster_groups,
        }


class KiloSortSortingExtractor(BasePhyKilosortSortingExtractor):
    """Load Kilosort format data as a sorting extractor.

    Parameters
    ----------
    folder_path : str or Path
        Path to the output Kilosort folder (containing the params.py).
    keep_good_only : bool, default: True
        Whether to only keep good units.
        If True, only Kilosort-labeled 'good' units are returned.
    remove_empty_units : bool, default: True
        If True, empty units are removed from the sorting extractor.

    Returns
    -------
    extractor : KiloSortSortingExtractor
        The loaded Sorting object.
    """

    def __init__(self, folder_path: Path | str, keep_good_only: bool = False, remove_empty_units: bool = True):
        BasePhyKilosortSortingExtractor.__init__(
            self,
            folder_path,
            exclude_cluster_groups=None,
            keep_good_only=keep_good_only,
            remove_empty_units=remove_empty_units,
        )

        self._kwargs = {"folder_path": str(Path(folder_path).absolute()), "keep_good_only": keep_good_only}


read_phy = define_function_from_class(source_class=PhySortingExtractor, name="read_phy")
read_kilosort = define_function_from_class(source_class=KiloSortSortingExtractor, name="read_kilosort")


def read_kilosort_as_analyzer(folder_path, unwhiten=True) -> SortingAnalyzer:
    """
    Load Kilosort output into a SortingAnalyzer. Output from Kilosort version 4.1 and
    above are supported. The function may work on older versions of Kilosort output,
    but these are not carefully tested. Please check your output carefully.

    Parameters
    ----------
    folder_path : str or Path
        Path to the output Phy folder (containing the params.py).
    unwhiten : bool, default: True
        Unwhiten the templates computed by kilosort.

    Returns
    -------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    """

    phy_path = Path(folder_path)

    sorting = read_phy(phy_path)
    sampling_frequency = sorting.sampling_frequency

    # kilosort occasionally contains a few spikes just beyond the recording end point, which can lead
    # to errors later. To avoid this, we pad the recording with an extra second of blank time.
    duration = sorting._sorting_segments[0]._all_spikes[-1] / sampling_frequency + 1

    if (phy_path / "probe.prb").is_file():
        probegroup = read_prb(phy_path / "probe.prb")
        if len(probegroup.probes) > 0:
            warnings.warn("Found more than one probe. Selecting the first probe in ProbeGroup.")
        probe = probegroup.probes[0]
    elif (phy_path / "channel_positions.npy").is_file():
        probe = Probe(si_units="um")
        channel_positions = np.load(phy_path / "channel_positions.npy")
        probe.set_contacts(channel_positions)
        probe.set_device_channel_indices(range(probe.get_contact_count()))
    else:
        AssertionError(f"Cannot read probe layout from folder {phy_path}.")

    # to make the initial analyzer, we'll use a fake recording and set it to None later
    recording, _ = generate_ground_truth_recording(
        probe=probe,
        sampling_frequency=sampling_frequency,
        durations=[duration],
        num_units=1,
        seed=1205,
    )

    sparsity = _make_sparsity_from_templates(sorting, recording, phy_path)

    sorting_analyzer = create_sorting_analyzer(sorting, recording, sparse=True, sparsity=sparsity)

    # first compute random spikes. These do nothing, but are needed for si-gui to run
    sorting_analyzer.compute("random_spikes")

    _make_templates(sorting_analyzer, phy_path, sparsity.mask, sampling_frequency, unwhiten=unwhiten)
    _make_locations(sorting_analyzer, phy_path)

    sorting_analyzer._recording = None
    return sorting_analyzer


def _make_locations(sorting_analyzer, kilosort_output_path):
    """Constructs a `spike_locations` extension from the amplitudes numpy array
    in `kilosort_output_path`, and attaches the extension to the `sorting_analyzer`."""

    locations_extension = ComputeSpikeLocations(sorting_analyzer)

    spike_locations_path = kilosort_output_path / "spike_positions.npy"
    if spike_locations_path.is_file():
        locs_np = np.load(spike_locations_path)
    else:
        return

    # Check that the spike locations vector is the same size as the spike vector
    num_spikes = len(sorting_analyzer.sorting.to_spike_vector())
    num_spike_locs = len(locs_np)
    if num_spikes != num_spike_locs:
        warnings.warn(
            "The number of spikes does not match the number of spike locations in `spike_positions.npy`. Skipping spike locations."
        )
        return

    num_dims = len(locs_np[0])
    column_names = ["x", "y", "z"][:num_dims]
    dtype = [(name, locs_np.dtype) for name in column_names]

    structured_array = np.zeros(len(locs_np), dtype=dtype)
    for coordinate_index, column_name in enumerate(column_names):
        structured_array[column_name] = locs_np[:, coordinate_index]

    locations_extension.data = {"spike_locations": structured_array}
    locations_extension.params = {}
    locations_extension.run_info = {"run_completed": True}

    sorting_analyzer.extensions["spike_locations"] = locations_extension


def _make_sparsity_from_templates(sorting, recording, kilosort_output_path):
    """Constructs the `ChannelSparsity` of from kilosort output, by seeing if the
    templates output is zero or not on all channels."""

    templates = np.load(kilosort_output_path / "templates.npy")

    unit_ids = sorting.unit_ids
    channel_ids = recording.channel_ids

    # The raw templates have dense dimensions (num chan)x(num samples)x(num units)
    # but are zero on many channels, which implicitly defines the sparsity
    mask = np.sum(np.abs(templates), axis=1) != 0
    return ChannelSparsity(mask, unit_ids=unit_ids, channel_ids=channel_ids)


def _make_templates(sorting_analyzer, kilosort_output_path, mask, sampling_frequency, unwhiten=True):
    """Constructs a `templates` extension from the amplitudes numpy array
    in `kilosort_output_path`, and attaches the extension to the `sorting_analyzer`."""

    template_extension = ComputeTemplates(sorting_analyzer)

    whitened_templates = np.load(kilosort_output_path / "templates.npy")
    wh_inv = np.load(kilosort_output_path / "whitening_mat_inv.npy")
    new_templates = _compute_unwhitened_templates(whitened_templates, wh_inv) if unwhiten else whitened_templates

    template_extension.data = {"average": new_templates}

    ops_path = kilosort_output_path / "ops.npy"
    if ops_path.is_file():
        ops = np.load(ops_path, allow_pickle=True)

        number_samples_before_template_peak = ops.item(0)["nt0min"]
        total_template_samples = ops.item(0)["nt"]

        number_samples_after_template_peak = total_template_samples - number_samples_before_template_peak

        ms_before = number_samples_before_template_peak / (sampling_frequency // 1000)
        ms_after = number_samples_after_template_peak / (sampling_frequency // 1000)

    # Used for kilosort 2, 2.5 and 3
    else:

        warnings.warn("Can't extract `ms_before` and `ms_after` from Kilosort output. Guessing a sensible value.")

        samples_in_templates = np.shape(new_templates)[1]
        template_extent_ms = (samples_in_templates + 1) / (sampling_frequency // 1000)
        ms_before = template_extent_ms / 3
        ms_after = 2 * template_extent_ms / 3

    params = {
        "operators": ["average"],
        "ms_before": ms_before,
        "ms_after": ms_after,
        "peak_sign": "both",
    }

    template_extension.params = params
    template_extension.run_info = {"run_completed": True}

    sorting_analyzer.extensions["templates"] = template_extension


def _compute_unwhitened_templates(whitened_templates, wh_inv):
    """Constructs unwhitened templates from whitened_templates, by
    applying an inverse whitening matrix."""

    # templates have dimension (num units) x (num samples) x (num channels)
    # whitening inverse has dimension (num units) x (num channels)
    # to undo whitening, we need do matrix multiplication on the channel index
    unwhitened_templates = np.einsum("ij,klj->kli", wh_inv, whitened_templates)

    return unwhitened_templates
