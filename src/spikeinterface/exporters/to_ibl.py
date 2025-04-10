from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from spikeinterface.core import ChannelSparsity, SortingAnalyzer
from spikeinterface.core.job_tools import divide_segment_into_chunks
from spikeinterface.core.template_tools import get_template_extremum_channel
from spikeinterface.exporters import (
    export_to_phy,
)


def export_to_ibl(
    analyzer: SortingAnalyzer,
    output_folder: str | Path,
    lfp_recording=None,
    rms_win_length_sec=3,
    welch_win_length_samples=2**14,
    total_secs_spec_dens=100,
    only_ibl_specific_steps=False,
    sparsity: Optional[ChannelSparsity] = None,
    remove_if_exists: bool = False,
    verbose: bool = True,
    **job_kwargs,
):
    """
    Exports a sorting analyzer to the IBL gui format (similar to the Phy format with some extras).

    Parameters
    ----------
    analyzer: SortingAnalyzer
        The sorting analyzer object to use for spike information. 
        Should also contain the pre-processed recording to use for AP-band data.
    output_folder: str | Path
        The output folder for the exports.
    lfp_recording: Any SI Recording object, default None
        The pre-processed recording to use for LFP data. If None, the LFP data is not exported.
    rms_win_length_sec: float, default: 3
        The window length in seconds for the RMS calculation (on the LFP data).
    welch_win_length_samples: int, default: 2^14
        The window length in samples for the Welch spectral density computation (on the LFP data).
    total_secs_spec_dens: int, default: 100
        The total number of seconds to use for the spectral density calculation.
    only_ibl_specific_steps: bool, default: False
        If True, only the IBL specific steps are run (i.e. skips calling `export_to_phy`)
    sparsity: ChannelSparsity or None, default: None
        The sparsity object (currently only respected for phy part of the export)
    remove_if_exists: bool, default: False
        If True and "output_folder" exists, it is removed and overwritten
    verbose: bool, default: True
        If True, output is verbose

    """

    try:
        from scipy.signal import welch
    except ImportError as e:
        raise ImportError(
            "Please install scipy to use the export_to_ibl function."
        ) from e

    # Output folder checks
    if isinstance(output_folder, str):
        output_folder = Path(output_folder)
    output_folder = Path(output_folder).absolute()
    if output_folder.is_dir():
        if remove_if_exists:
            shutil.rmtree(output_folder)
        else:
            raise FileExistsError(f"{output_folder} already exists")
    else:
        pass
        # don't make the output dir yet, b/c export_to_phy will do that for us.

    if verbose:
        print("Exporting recording to IBL format...")

    # Compute any missing extensions
    available_extension_names = analyzer.get_saved_extension_names()
    required_exts = [
        "templates",
        "template_similarity",
        "spike_locations",
        "noise_levels",
        "quality_metrics",
    ]
    required_qms = ["amplitude_median", "isi_violations_ratio", "amplitude_cutoff"]
    for ext in required_exts:
        if ext not in available_extension_names:
            if ext == "quality_metrics":
                kwargs = {"skip_pc_metrics": False}
            else:
                kwargs = {}
            analyzer.compute(ext, verbose=verbose, **kwargs)
        elif ext == "quality_metrics":
            qm = analyzer.get_extension("quality_metrics").get_data()
            for rqm in required_qms:
                if rqm not in qm:
                    analyzer.compute(
                        "quality_metrics",
                        metric_names=[rqm],
                        verbose=verbose,
                    )

    # # Start by just exporting to phy
    if not only_ibl_specific_steps:
        if verbose:
            print("Doing phy-like export...")
        export_to_phy(
            analyzer,
            output_folder,
            compute_amplitudes=True,
            compute_pc_features=False,
            sparsity=sparsity,
            copy_binary=False,
            template_mode="median",
            verbose=verbose,
            use_relative_path=False,
            **job_kwargs,
        )

    # Make sure output dir exists, in case user skips export_to_phy
    if not output_folder.is_dir():
        os.makedirs(output_folder)

    if verbose:
        print("Running IBL-specific steps...")

    # Now we need to add the extra IBL specific files
    # See here for docs on the format: https://github.com/int-brain-lab/iblapps/wiki/3.-Overview-of-datasets#input-histology-data

    # Subset channels in case some were excluded from spike sorting
    (channel_inds,) = np.isin(
        analyzer.recording.channel_ids, analyzer.channel_ids
    ).nonzero()

    # TODO: put this into a chunk extractor
    def _get_rms(rec):
        chunk_nframes = int(rms_win_length_sec * rec.sampling_frequency)
        chunks = divide_segment_into_chunks(rec.get_num_samples(), chunk_nframes)
        chunk_rms = np.zeros((len(chunks), rec.get_num_channels()))
        chunk_start_times = np.zeros((len(chunks),))
        for iChunk, (start_frame, stop_frame) in enumerate(tqdm(chunks)):
            traces = rec.get_traces(start_frame=start_frame, end_frame=stop_frame)
            chunk_rms[iChunk, :] = np.sqrt(np.mean(traces**2, axis=0))
            chunk_start_times[iChunk] = start_frame / rec.sampling_frequency
        chunk_rms = chunk_rms[:, channel_inds]
        chunk_rms = chunk_rms.astype(np.float32)
        chunk_start_times = chunk_start_times.astype(np.float32)
        return chunk_rms, chunk_start_times

    # Get RMS for the AP data. We will use a window of length rms_win_length_sec seconds slid over the entire recording.
    ap_rec = analyzer.recording
    if ap_rec.get_num_segments() != 1:
        warnings.warn(
            "Found ap recording with more than one segment, only using initial segment."
        )
        ap_rec = ap_rec[0]
    chunk_rms, chunk_start_times = _get_rms(ap_rec)
    np.save(os.path.join(output_folder, "_iblqc_ephysTimeRmsAP.rms.npy"), chunk_rms)
    np.save(
        os.path.join(output_folder, "_iblqc_ephysTimeRmsAP.timestamps.npy"),
        chunk_start_times,
    )

    if lfp_recording is not None:
        # Get RMS for the LFP data.
        if lfp_recording.get_num_segments() != 1:
            warnings.warn(
                "Found lfp recording with more than one segment, only using initial segment."
            )
            lfp_recording = lfp_recording[0]
        chunk_rms, chunk_start_times = _get_rms(lfp_recording)
        np.save(os.path.join(output_folder, "_iblqc_ephysTimeRmsLF.rms.npy"), chunk_rms)
        np.save(
            os.path.join(output_folder, "_iblqc_ephysTimeRmsLF.timestamps.npy"),
            chunk_start_times,
        )

        # Get spectral density on a snippet of LFP data
        end_frame = int(total_secs_spec_dens * lfp_recording.sampling_frequency)
        traces = lfp_recording.get_traces(
            start_frame=0, end_frame=end_frame
        )  # time x channels
        spec_density = np.zeros((welch_win_length_samples // 2 + 1, traces.shape[1]))
        for iCh in range(traces.shape[1]):
            f, Pxx = welch(
                traces[:, iCh],
                fs=lfp_recording.sampling_frequency,
                nperseg=welch_win_length_samples,
            )
            spec_density[:, iCh] = Pxx
        spec_density = spec_density[
            :, channel_inds
        ]  # only keep channels that were used for spike sorting
        spec_density = spec_density.astype(np.float32)
        f = f.astype(np.float32)
        assert spec_density.shape[0] == len(f)
        np.save(
            os.path.join(output_folder, "_iblqc_ephysSpectralDensityLF.power.npy"),
            spec_density,
        )
        np.save(
            os.path.join(output_folder, "_iblqc_ephysSpectralDensityLF.freqs.npy"), f
        )

    ### Save spike info ###

    spike_locations = analyzer.load_extension("spike_locations").get_data()
    spike_depths = spike_locations["y"]

    # convert clusters and squeeze
    clusters = np.load(output_folder / "spike_clusters.npy")
    np.save(output_folder / "spike_clusters.npy", np.squeeze(clusters.astype("uint32")))

    # convert times and squeeze
    times = np.load(output_folder / "spike_times.npy")
    np.save(
        output_folder / "spike_times.npy", np.squeeze(times / 30000.0).astype("float64")
    )

    # convert amplitudes and squeeze
    amps = np.load(output_folder / "amplitudes.npy")
    np.save(output_folder / "amplitudes.npy", np.squeeze(-amps / 1e6).astype("float64"))

    # save depths and channel inds
    np.save(output_folder / "spike_depths.npy", spike_depths)
    np.save(
        output_folder / "channel_inds.npy", np.arange(len(channel_inds), dtype="int")
    )

    # # save templates
    cluster_channels = []
    cluster_peakToTrough = []
    cluster_waveforms = []
    templates = analyzer.get_extension("templates").get_data()
    extremum_channel_indices = get_template_extremum_channel(analyzer, outputs="index")

    for unit_idx, unit_id in enumerate(analyzer.unit_ids):
        waveform = templates[unit_idx, :, :]
        extremum_channel_index = extremum_channel_indices[unit_id]
        peak_waveform = waveform[:, extremum_channel_index]
        peakToTrough = (
            np.argmax(peak_waveform) - np.argmin(peak_waveform)
        ) / analyzer.sampling_frequency
        # cluster_channels.append(int(channel_locs[extremum_channel_index, 1] / 10)) # ??? fails for odd nums of units
        cluster_channels.append(
            extremum_channel_index
        )  # see: https://github.com/SpikeInterface/spikeinterface/issues/2843#issuecomment-2148164870
        cluster_peakToTrough.append(peakToTrough)
        cluster_waveforms.append(waveform)

    np.save(output_folder / "cluster_peakToTrough.npy", np.array(cluster_peakToTrough))
    np.save(output_folder / "cluster_waveforms.npy", np.stack(cluster_waveforms))
    np.save(output_folder / "cluster_channels.npy", np.array(cluster_channels))

    # rename files from this func and the phy export func
    _FILE_RENAMES = [  # file_in, file_out
        ("channel_positions.npy", "channels.localCoordinates.npy"),
        ("channel_inds.npy", "channels.rawInd.npy"),
        ("cluster_peakToTrough.npy", "clusters.peakToTrough.npy"),
        ("cluster_channels.npy", "clusters.channels.npy"),
        ("cluster_waveforms.npy", "clusters.waveforms.npy"),
        ("spike_clusters.npy", "spikes.clusters.npy"),
        ("amplitudes.npy", "spikes.amps.npy"),
        ("spike_depths.npy", "spikes.depths.npy"),
        ("spike_times.npy", "spikes.times.npy"),
    ]

    for names in _FILE_RENAMES:
        old_name = output_folder / names[0]
        new_name = output_folder / names[1]
        os.rename(old_name, new_name)

    # save quality metrics
    qm = analyzer.load_extension("quality_metrics")
    qm_data = qm.get_data()
    qm_data.index.name = "cluster_id"
    qm_data["cluster_id.1"] = qm_data.index.values
    good_ibl = (  # rough estimate of ibl standards
        (qm_data["amplitude_median"] > 50)
        & (qm_data["isi_violations_ratio"] < 0.2)
        & (qm_data["amplitude_cutoff"] < 0.1)
    )
    qm_data["label"] = good_ibl.astype("int")
    qm_data.to_csv(output_folder / "clusters.metrics.csv")
