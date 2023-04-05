from pathlib import Path
import copy
from packaging import version

from ..basesorter import BaseSorter
from spikeinterface.core.old_api_utils import NewToOldRecording

from spikeinterface.core import load_extractor
from spikeinterface.extractors import HerdingspikesSortingExtractor


class HerdingspikesSorter(BaseSorter):
    """HerdingSpikes Sorter object."""

    sorter_name = 'herdingspikes'

    requires_locations = True
    compatible_with_parallel = {'loky': True, 'multiprocessing': True, 'threading': False}
    _default_params = {
        # core params
        'clustering_bandwidth': 5.5,  # 5.0,
        'clustering_alpha': 5.5,  # 5.0,
        'clustering_n_jobs': -1,
        'clustering_bin_seeding': True,
        'clustering_min_bin_freq': 16,  # 10,
        'clustering_subset': None,
        'left_cutout_time': 0.3,  # 0.2,
        'right_cutout_time': 1.8,  # 0.8,
        'detect_threshold': 20,  # 24, #15,

        # extra probe params
        'probe_masked_channels': [],
        'probe_inner_radius': 70,
        'probe_neighbor_radius': 90,
        'probe_event_length': 0.26,
        'probe_peak_jitter': 0.2,

        # extra detection params
        't_inc': 100000,
        'num_com_centers': 1,
        'maa': 12,
        'ahpthr': 11,
        'out_file_name': "HS2_detected",
        'decay_filtering': False,
        'save_all': False,
        'amp_evaluation_time': 0.4,  # 0.14,
        'spk_evaluation_time': 1.0,

        # extra pca params
        'pca_ncomponents': 2,
        'pca_whiten': True,

        # bandpass filter
        'freq_min': 300.0,
        'freq_max': 6000.0,
        'filter': True,

        # rescale traces
        'pre_scale': True,
        'pre_scale_value': 20.0,

        # remove duplicates (based on spk_evaluation_time)
        'filter_duplicates': True
    }

    _params_description = {
        # core params
        'clustering_bandwidth': "Meanshift bandwidth, average spatial extent of spike clusters (um)",
        'clustering_alpha': "Scalar for the waveform PC features when clustering.",
        'clustering_n_jobs': "Number of cores to use for clustering.",
        'clustering_bin_seeding': "Enable clustering bin seeding.",
        'clustering_min_bin_freq': "Minimum spikes per bin for bin seeding.",
        'clustering_subset': "Number of spikes used to build clusters. All by default.",
        'left_cutout_time': "Cutout size before peak (ms).",
        'right_cutout_time': "Cutout size after peak (ms).",
        'detect_threshold': "Detection threshold",

        # extra probe params
        'probe_masked_channels': "Masked channels",
        'probe_inner_radius': "Radius of area around probe channel for localization",
        'probe_neighbor_radius': "Radius of area around probe channel for neighbor classification.",
        'probe_event_length': "Duration of a spike event (ms)",
        'probe_peak_jitter': "Maximum peak misalignment for synchronous spike (ms)",

        # extra detection params
        't_inc': "Number of samples per chunk during detection.",
        'num_com_centers': "Number of centroids to average when localizing.",
        'maa': "Minimum summed spike amplitude for spike acceptance.",
        'ahpthr': "Requires magnitude of spike rebound for acceptance",
        'out_file_name': "File name for storage of unclustered detected spikes",
        'decay_filtering': "Experimental: Set to True at your risk",
        'save_all': "Save all working files after sorting (slow)",
        'amp_evaluation_time': "Amplitude evaluation time (ms)",
        'spk_evaluation_time': "Spike evaluation time (ms)",

        # extra pca params
        'pca_ncomponents': "Number of principal components to use when clustering",
        'pca_whiten': "If true, whiten data for pca",

        # bandpass filter
        'freq_min': "High-pass filter cutoff frequency",
        'freq_max': "Low-pass filter cutoff frequency",
        'filter': "Enable or disable filter",

        # rescale traces
        'pre_scale': "Scales recording traces to optimize HerdingSpikes performance",
        'pre_scale_value': "Scale to apply in case of pre-scaling of traces",

        # remove duplicates (based on spk_evaluation_time)
        'filter_duplicates': "Remove spike duplicates (based on spk_evaluation_time)"
    }

    sorter_description = """Herding Spikes is a density-based spike sorter designed for high-density retinal recordings.
    It uses both PCA features and an estimate of the spike location to cluster different units.
    For more information see https://doi.org/10.1016/j.jneumeth.2016.06.006"""

    installation_mesg = """\nTo use HerdingSpikes run:\n
       >>> pip install herdingspikes
    More information on HerdingSpikes at:
      * https://github.com/mhhennig/hs2
    """

    handle_multi_segment = False

    @classmethod
    def is_installed(cls):
        try:
            import herdingspikes as hs
            HAVE_HS = True
        except ImportError:
            HAVE_HS = False
        return HAVE_HS

    @classmethod
    def get_sorter_version(cls):
        import herdingspikes as hs
        return hs.__version__

    @classmethod
    def _check_apply_filter_in_params(cls, params):
        return params['filter']

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        # nothing to copy inside the folder : Herdingspikes used natively spikeinterface
        pass

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        import herdingspikes as hs
        from spikeinterface.preprocessing import bandpass_filter, normalize_by_quantile

        hs_version = version.parse(hs.__version__)

        if hs_version >= version.parse("0.3.99"):
            new_api = True
        else:
            new_api = False

        recording = load_extractor(sorter_output_folder.parent / 'spikeinterface_recording.json')

        p = params

        # Bandpass filter
        if p['filter'] and p['freq_min'] is not None and p['freq_max'] is not None:
            recording = bandpass_filter(
                recording=recording, freq_min=p['freq_min'], freq_max=p['freq_max'])

        if p['pre_scale']:
            recording = normalize_by_quantile(
                recording=recording, scale=p['pre_scale_value'],
                median=0.0, q1=0.05, q2=0.95
            )

        if new_api:
            recording_to_hs = recording
        else:
            print('herdingspikes version<0.3.99 uses the OLD spikeextractors with NewToOldRecording.\n'
                  'Consider updating herdingspikes (pip install herdingspikes>=0.3.99)')
            recording_to_hs = NewToOldRecording(recording)

        # this should have its name changed
        Probe = hs.probe.RecordingExtractor(
            recording_to_hs,
            masked_channels=p['probe_masked_channels'],
            inner_radius=p['probe_inner_radius'],
            neighbor_radius=p['probe_neighbor_radius'],
            event_length=p['probe_event_length'],
            peak_jitter=p['probe_peak_jitter'])

        H = hs.HSDetection(
            Probe, file_directory_name=str(sorter_output_folder),
            left_cutout_time=p['left_cutout_time'],
            right_cutout_time=p['right_cutout_time'],
            threshold=p['detect_threshold'],
            to_localize=True,
            num_com_centers=p['num_com_centers'],
            maa=p['maa'],
            ahpthr=p['ahpthr'],
            out_file_name=p['out_file_name'],
            decay_filtering=p['decay_filtering'],
            save_all=p['save_all'],
            amp_evaluation_time=p['amp_evaluation_time'],
            spk_evaluation_time=p['spk_evaluation_time']
        )

        H.DetectFromRaw(load=True, tInc=int(p['t_inc']))

        sorted_file = str(sorter_output_folder / 'HS2_sorted.hdf5')
        if (not H.spikes.empty):
            C = hs.HSClustering(H)
            C.ShapePCA(pca_ncomponents=p['pca_ncomponents'],
                       pca_whiten=p['pca_whiten'])
            C.CombinedClustering(
                alpha=p['clustering_alpha'],
                cluster_subset=p['clustering_subset'],
                bandwidth=p['clustering_bandwidth'],
                bin_seeding=p['clustering_bin_seeding'],
                n_jobs=p['clustering_n_jobs'],
                min_bin_freq=p['clustering_min_bin_freq']
            )
        else:
            C = hs.HSClustering(H)

        if p['filter_duplicates']:
            uids = C.spikes.cl.unique()
            for u in uids:
                s = C.spikes[C.spikes.cl == u].t.diff() < p['spk_evaluation_time'] / 1000 * Probe.fps
                C.spikes = C.spikes.drop(s.index[s])

        if verbose:
            print('Saving to', sorted_file)
        C.SaveHDF5(sorted_file, sampling=Probe.fps)

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        return HerdingspikesSortingExtractor(file_path=Path(sorter_output_folder) / 'HS2_sorted.hdf5',
                                             load_unit_info=True)
