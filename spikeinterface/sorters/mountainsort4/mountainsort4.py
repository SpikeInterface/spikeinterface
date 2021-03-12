import copy
from pathlib import Path


# TODO
#~ from spiketoolkit.preprocessing import bandpass_filter, whiten

from ..basesorter import BaseSorter
from spikeinterface.core import load_extractor

# TODO
# from spikeinterface.extractors import MdaSortingExtractor
from spikeinterface.extractors import NpzSortingExtractor

try:
    import ml_ms4alg

    HAVE_MS4 = True
except ImportError:
    HAVE_MS4 = False


class Mountainsort4Sorter(BaseSorter):
    """
    Mountainsort
    """

    sorter_name = 'mountainsort4'
    requires_locations = False
    compatible_with_parallel = {'loky': True, 'multiprocessing': False, 'threading': False}

    _default_params = {
        'detect_sign': -1,  # Use -1, 0, or 1, depending on the sign of the spikes in the recording
        'adjacency_radius': -1,  # Use -1 to include all channels in every neighborhood
        'freq_min': 300,  # Use None for no bandpass filtering
        'freq_max': 6000,
        'filter': True,
        'whiten': True,  # Whether to do channel whitening as part of preprocessing
        'curation': False,
        'num_workers': None,
        'clip_size': 50,
        'detect_threshold': 3,
        'detect_interval': 10,  # Minimum number of timepoints between events detected on the same channel
        'noise_overlap_threshold': 0.15,  # Use None for no automated curation'
    }

    _params_description = {
        'detect_sign': "Use -1 (negative) or 1 (positive) depending "
                       "on the sign of the spikes in the recording",  # Use -1, 0, or 1, depending on the sign of the spikes in the recording
        'adjacency_radius': "Radius in um to build channel neighborhood "
                            "(Use -1 to include all channels in every neighborhood)",  # Use -1 to include all channels in every neighborhood
        'freq_min': "High-pass filter cutoff frequency",
        'freq_max': "Low-pass filter cutoff frequency",
        'filter': "Enable or disable filter",
        'whiten': "Enable or disable whitening",
        'curation': "Enable or disable curation",
        'num_workers': "Number of workers (if None, half of the cpu number is used)",
        'clip_size': "Number of samples per waveform",
        'detect_threshold': "Threshold for spike detection",
        'detect_interval': "Minimum number of timepoints between events detected on the same channel",
        'noise_overlap_threshold': "Noise overlap threshold for automatic curation",
    }

    sorter_description = """Mountainsort4 is a fully automatic density-based spike sorter using the isosplit clustering 
    method and automatic curation procedures. For more information see https://doi.org/10.1016/j.neuron.2017.08.030"""

    installation_mesg = """\nTo use Mountainsort4 run:\n
       >>> pip install ml_ms4alg

    More information on mountainsort at:
      * https://github.com/flatironinstitute/mountainsort
    """

    #~ def __init__(self, **kargs):
        #~ BaseSorter.__init__(self, **kargs)
    
    @classmethod
    def is_installed(cls):
        return HAVE_MS4
    
    @staticmethod
    def get_sorter_version():
        if hasattr(ml_ms4alg, '__version__'):
            return ml_ms4alg.__version__
        return 'unknown'

    @classmethod
    def _check_apply_filter_in_params(cls, params):
        return params['filter']

    @classmethod
    def _setup_recording(cls, recording, output_folder, params, verbose):
        pass

    @classmethod
    def _run_from_folder(cls, output_folder, params, verbose):
        raise NotImplementedError('Mountainsort4 need to use the new get_traces() transpose + segment_index!!!')
        
        recording = load_extractor(output_folder / 'spikeinterface_recording.json')

        # alias to params
        p = params

        samplerate = recording.get_sampling_frequency()

        # Bandpass filter
        if p['filter'] and p['freq_min'] is not None and p['freq_max'] is not None:
            recording = bandpass_filter(recording=recording, freq_min=p['freq_min'], freq_max=p['freq_max'])

        # Whiten
        if p['whiten']:
            recording = whiten(recording=recording)

        # Check location no more needed done in basesorter
        sorting = ml_ms4alg.mountainsort4(
            recording=recording,
            detect_sign=p['detect_sign'],
            adjacency_radius=p['adjacency_radius'],
            clip_size=p['clip_size'],
            detect_threshold=p['detect_threshold'],
            detect_interval=p['detect_interval'],
            num_workers=p['num_workers'],
            verbose=verbose
        )

        # Curate
        if p['noise_overlap_threshold'] is not None and p['curation'] is True:
            if verbose:
                print('Curating')
            sorting = ml_ms4alg.mountainsort4_curation(
                recording=recording,
                sorting=sorting,
                noise_overlap_threshold=p['noise_overlap_threshold']
            )

        # MdaSortingExtractor.write_sorting(sorting, str(output_folder / 'firings.mda'))
        # samplerate_fname = str(output_folder / 'samplerate.txt')
        # with open(samplerate_fname, 'w') as f:
            # f.write('{}'.format(samplerate))
     
        NpzSortingExtractor.write_sorting(sorting, str(output_folder / 'firings.npz'))

   
    @classmethod
    def get_result_from_folder(cls, output_folder):
        output_folder = Path(output_folder)
        
        #~ tmpdir = output_folder
        #~ result_fname = str(tmpdir / 'firings.mda')
        #~ samplerate_fname = str(tmpdir / 'samplerate.txt')
        #~ with open(samplerate_fname, 'r') as f:
            #~ samplerate = float(f.read())
        #~  sorting = MdaSortingExtractor(file_path=result_fname, sampling_frequency=samplerate)
        
        result_fname = output_folder/ 'firings.npz'
        sorting = NpzSortingExtractor.write_sorting(result_fname)
        
        return sorting
