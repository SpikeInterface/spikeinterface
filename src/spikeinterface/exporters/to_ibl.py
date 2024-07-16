from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional
import shutil

import numpy as np
import numpy.typing as npt
from scipy.signal import welch
from tqdm.auto import tqdm

from spikeinterface.core import (
    BinaryFolderRecording,
    BinaryRecordingExtractor,
    ChannelSparsity,
    WaveformExtractor,
)
from spikeinterface.core.template_tools import get_template_extremum_channel
from spikeinterface.exporters import (
    export_to_phy,
)
from spikeinterface.exporters.to_ibl_utils import (
    WindowGenerator,
    fscale,
    hp,
    rms,
    save_object_npy,
)

def export_to_ibl(
    recording: BinaryRecordingExtractor | BinaryFolderRecording,
    waveform_extractor: WaveformExtractor,
    output_folder: str | Path,
    rms_win_length_sec = 3,
    welch_win_length_samples = 1024,
    total_secs = 100,
    only_ibl_specific_steps=False,
    compute_pc_features: bool = True,
    compute_amplitudes: bool = True,
    sparsity: Optional[ChannelSparsity] = None,
    copy_binary: bool = True,
    remove_if_exists: bool = False,
    peak_sign: Literal["both", "neg", "pos"] = "neg",
    template_mode: str = "median",
    dtype: Optional[npt.DTypeLike] = None,
    verbose: bool = True,
    use_relative_path: bool = False,
    **job_kwargs,
):
    """
    Exports a waveform extractor to the IBL gui format (similar to the Phy format with some extras).

    Parameters
    ----------
    recording: BinaryRecordingExtractor | BinaryFolderRecording
        The recording extractor or the recording folder.
    waveform_extractor: a WaveformExtractor or None
        If WaveformExtractor is provide then the compute is faster otherwise [?].
    output_folder: str | Path
        The output folder where the phy template-gui files are saved
    rms_win_length_sec: float, default: 3
        The window length in seconds for the RMS calculation.
    welch_win_length_samples: int, default: 1024
        The window length in samples for the Welch method.
    total_secs: int, default: 100
        The total number of seconds to use for the spectral density calculation.
    only_ibl_specific_steps: bool, default: False
        If True, only the IBL specific steps are run (i.e. skips calling `export_to_phy`)
    compute_pc_features: bool, default: True
        If True, pc features are computed
    compute_amplitudes: bool, default: True
        If True, waveforms amplitudes are computed
    sparsity: ChannelSparsity or None, default: None
        The sparsity object
    copy_binary: bool, default: True
        If True, the recording is copied and saved in the phy "output_folder"
    remove_if_exists: bool, default: False
        If True and "output_folder" exists, it is removed and overwritten
    peak_sign: "neg" | "pos" | "both", default: "neg"
        Used by compute_spike_amplitudes
    template_mode: str, default: "median"
        Parameter "mode" to be given to WaveformExtractor.get_template()
    dtype: dtype or None, default: None
        Dtype to save binary data
    verbose: bool, default: True
        If True, output is verbose
    use_relative_path : bool, default: False
        If True and `copy_binary=True` saves the binary file `dat_path` in the `params.py` relative to `output_folder` (ie `dat_path=r"recording.dat"`). If `copy_binary=False`, then uses a path relative to the `output_folder`
        If False, uses an absolute path in the `params.py` (ie `dat_path=r"path/to/the/recording.dat"`)
    {}

    """

    if isinstance(output_folder, str):
        output_folder = Path(output_folder)

    output_folder = Path(output_folder).absolute()
    if output_folder.is_dir():
        if remove_if_exists:
            shutil.rmtree(output_folder)
        else:
            raise FileExistsError(f"{output_folder} already exists")

    # Start by just exporting to phy
    if verbose: 
        print("Exporting recording to IBL format...")

    if not only_ibl_specific_steps:
        if verbose: 
            print("Doing phy-like export...")
        export_to_phy(
            waveform_extractor,
            output_folder,
            compute_amplitudes=compute_amplitudes,
            compute_pc_features=compute_pc_features,
            sparsity=sparsity,
            copy_binary=copy_binary,
            remove_if_exists=remove_if_exists,
            peak_sign=peak_sign,
            template_mode=template_mode,
            dtype=dtype,
            verbose=verbose,
            use_relative_path=use_relative_path,
            **job_kwargs,
        )

    if verbose:
        print("Running IBL-specific steps...")

    # Now we need to add the extra IBL specific files
    (channel_inds,) = np.isin(
        recording.channel_ids, waveform_extractor.channel_ids
    ).nonzero()

    ### Run spectral density and rms ###
    fs_ap = recording.sampling_frequency
    rms_win_length_samples_ap = 2 ** np.ceil(np.log2(fs_ap * rms_win_length_sec))
    total_samples_ap = int(np.min([fs_ap * total_secs, recording.get_num_samples()]))

    # the window generator will generates window indices
    wingen = WindowGenerator(
        ns=total_samples_ap, nswin=rms_win_length_samples_ap, overlap=0
    )
    win = {
        "TRMS": np.zeros((wingen.nwin, recording.get_num_channels())),
        "nsamples": np.zeros((wingen.nwin,)),
        "fscale": fscale(welch_win_length_samples, 1 / fs_ap, one_sided=True),
        "tscale": wingen.tscale(fs=fs_ap),
    }
    win["spectral_density"] = np.zeros(
        (len(win["fscale"]), recording.get_num_channels())
    )

    # @Josh: this could be dramatically sped up if we employ SpikeInterface parallelization
    with tqdm(total=wingen.nwin) as pbar:
        for first, last in wingen.firstlast:
            D = recording.get_traces(start_frame=first, end_frame=last).T
            # remove low frequency noise below 1 Hz
            D = hp(D, 1 / fs_ap, [0, 1])
            iw = wingen.iw
            win["TRMS"][iw, :] = rms(D)
            win["nsamples"][iw] = D.shape[1]

            # the last window may be smaller than what is needed for welch
            if last - first < welch_win_length_samples:
                continue

            # compute a smoothed spectrum using welch method
            _, w = welch(
                D,
                fs=fs_ap,
                window="hann",
                nperseg=welch_win_length_samples,
                detrend="constant",
                return_onesided=True,
                scaling="density",
                axis=-1,
            )
            win["spectral_density"] += w.T
            # print at least every 20 windows
            if (iw % min(20, max(int(np.floor(wingen.nwin / 75)), 1))) == 0:
                pbar.update(iw)

    win["TRMS"] = win["TRMS"][:, channel_inds]
    win["spectral_density"] = win["spectral_density"][:, channel_inds]

    alf_object_time = "ephysTimeRmsAP"
    alf_object_freq = "ephysSpectralDensityAP"

    tdict = {
        "rms": win["TRMS"].astype(np.single),
        "timestamps": win["tscale"].astype(np.single),
    }
    save_object_npy(
        output_folder, object=alf_object_time, dico=tdict, namespace="iblqc"
    )

    fdict = {
        "power": win["spectral_density"].astype(np.single),
        "freqs": win["fscale"].astype(np.single),
    }
    save_object_npy(
        output_folder, object=alf_object_freq, dico=fdict, namespace="iblqc"
    )

    ### Save spike info ###

    # Confirm spike locations are available
    available_extension_names = waveform_extractor.get_available_extension_names()
    if "spike_locations" not in available_extension_names:
        from spikeinterface.postprocessing import compute_spike_locations

        compute_spike_locations(
            waveform_extractor, verbose=verbose
        )  # this should auto-save it

    spike_locations = waveform_extractor.load_extension("spike_locations").get_data()
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
    # num_chans = []

    templates = waveform_extractor.get_all_templates()
    # channel_locs = waveform_extractor.get_channel_locations()
    extremum_channel_indices = get_template_extremum_channel(
        waveform_extractor, outputs="index"
    )

    for unit_idx, unit_id in enumerate(waveform_extractor.unit_ids):
        waveform = templates[unit_idx, :, :]
        extremum_channel_index = extremum_channel_indices[unit_id]
        peak_waveform = waveform[:, extremum_channel_index]
        peakToTrough = (
            np.argmax(peak_waveform) - np.argmin(peak_waveform)
        ) / waveform_extractor.sampling_frequency
        # cluster_channels.append(int(channel_locs[extremum_channel_index, 1] / 10)) # ??? fails for odd nums of units
        cluster_channels.append(extremum_channel_index)  # see: https://github.com/SpikeInterface/spikeinterface/issues/2843#issuecomment-2148164870
        cluster_peakToTrough.append(peakToTrough)
        cluster_waveforms.append(waveform)

    np.save(output_folder / "cluster_peakToTrough.npy", np.array(cluster_peakToTrough))
    np.save(output_folder / "cluster_waveforms.npy", np.stack(cluster_waveforms))
    np.save(output_folder / "cluster_channels.npy", np.array(cluster_channels))

    # rename files
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
        # shutil.copyfile(old_name, new_name)
        os.rename(old_name, new_name)

    # save quality metrics
    qm = waveform_extractor.load_extension("quality_metrics")
    qm_data = qm.get_data()

    qm_data.index.name = "cluster_id"
    qm_data["cluster_id.1"] = qm_data.index.values

    qm_data.to_csv(output_folder / "clusters.metrics.csv")


# if __name__ == "__main__":

#     print("Running test script...")
#     rec = load_extractor("/n/groups/datta/Jonah/20231003_vlPAG_npx/raw_data/J04501/20240405_J04501/2024-04-05_18-46-54/preprocess")
#     we = load_waveforms("/n/groups/datta/Jonah/20231003_vlPAG_npx/raw_data/J04501/20240405_J04501/2024-04-05_18-46-54/kilosort4_clitest_preCompTemplates/waveforms_folder")
#     output_folder = "/n/groups/datta/Jonah/20231003_vlPAG_npx/raw_data/J04501/20240405_J04501/2024-04-05_18-46-54/ibl_exported"

#     # rec = load_extractor("/n/groups/datta/Jonah/20231003_vlPAG_npx/raw_data/J04501/20240403_J04501/2024-04-03_16-13-26/preprocess")
#     # we = load_waveforms("/n/groups/datta/Jonah/20231003_vlPAG_npx/raw_data/J04501/20240403_J04501/2024-04-03_16-13-26/kilosort4_clitest_preCompTemplates/waveforms_folder")
#     # output_folder = "/n/groups/datta/Jonah/20231003_vlPAG_npx/raw_data/J04501/20240403_J04501/2024-04-03_16-13-26/ibl_exported"

#     export_to_ibl(rec, we, output_folder, compute_pc_features=False, copy_binary=False)
#     print("Done!")
