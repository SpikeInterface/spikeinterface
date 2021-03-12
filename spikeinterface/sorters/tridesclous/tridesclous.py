from pathlib import Path
import os
import shutil
import numpy as np
import copy
import time
from pprint import pprint

import distutils.version


#~ from spikeinterface.extractors import TridesclousSortingExtractor

from ..basesorter import BaseSorter
from spikeinterface.core import BinaryRecordingExtractor

from probeinterface import write_prb

try:
    import tridesclous as tdc

    HAVE_TDC = True
except ImportError:
    HAVE_TDC = False


class TridesclousSorter(BaseSorter):
    """
    tridesclous is one of the more convinient, fast and elegant
    spike sorter.
    Everyone should test it.
    """

    sorter_name = 'tridesclous'
    requires_locations = False
    compatible_with_parallel = {'loky': True, 'multiprocessing': False, 'threading': False}

    _default_params = {
        'freq_min': 400.,
        'freq_max': 5000.,
        'detect_sign': -1,
        'detect_threshold': 5,
        'peak_span_ms': 0.7,
        'wf_left_ms': -2.0,
        'wf_right_ms': 3.0,
        'feature_method': 'auto',  # peak_max/global_pca/by_channel_pca
        'cluster_method': 'auto',  # pruningshears/dbscan/kmeans
        'clean_catalogue_gui': False,
    }

    _params_description = {
        'freq_min': "High-pass filter cutoff frequency",
        'freq_max': "Low-pass filter cutoff frequency",
        'detect_threshold': "Threshold for spike detection",
        'detect_sign': "Use -1 (negative) or 1 (positive) depending "
                       "on the sign of the spikes in the recording",
        'peak_span_ms': "Span of the peak in ms",
        'wf_left_ms': "Cut out before peak in ms",
        'wf_right_ms': " Cut out after peak in ms",
        'feature_method': "Feature method to use",  # peak_max/global_pca/by_channel_pca
        'cluster_method': "Feature method to use",  # pruningshears/dbscan/kmeans
        'clean_catalogue_gui': "Enable or disable interactive GUI for cleaning templates before peeler",
    }

    sorter_description = """Tridesclous is a template-matching spike sorter with a real-time engine. 
    For more information see https://tridesclous.readthedocs.io"""

    installation_mesg = """\nTo use Tridesclous run:\n
       >>> pip install tridesclous

    More information on tridesclous at:
      * https://github.com/tridesclous/tridesclous
      * https://tridesclous.readthedocs.io
    """
    
    # TODO make the TDC handle multi segment (should be easy)
    handle_multi_segment = False

    #~ def __init__(self, **kargs):
        #~ BaseSorter.__init__(self, **kargs)
    
    @classmethod
    def is_installed(cls):
        return HAVE_TDC
        
    @classmethod
    def get_sorter_version(cls):
        return tdc.__version__

    @classmethod
    def _check_params(cls, recording, output_folder, params):
        return params

    @classmethod
    def _setup_recording(cls, recording, output_folder, params, verbose):
        # reset the output folder
        #~ if output_folder.is_dir():
            #~ shutil.rmtree(str(output_folder))
        #~ os.makedirs(str(output_folder))

        # save prb file
        probegroup = recording.get_probegroup()
        prb_file = output_folder / 'probe.prb'
        write_prb(prb_file, probegroup)

        # source file
        # TODO wrap BinaryRecordingExtractor into neo source
        #~ if isinstance(recording, se.BinDatRecordingExtractor) and recording._time_axis == 0:
            #~ # no need to copy
            #~ raw_filename = recording._datfile
            #~ dtype = recording._timeseries.dtype.str
            #~ nb_chan = len(recording._channels)
            #~ offset = recording._timeseries.offset
        #~ else:
            #~ if self.verbose:
                #~ print('Local copy of recording')
            #~ # save binary file (chunk by hcunk) into a new file
            #~ raw_filename = output_folder / 'raw_signals.raw'
            #~ recording.write_to_binary_dat_format(raw_filename, time_axis=0, dtype='float32', chunk_mb=500)
            #~ dtype = 'float32'
            #~ offset = 0
        if verbose:
            print('Local copy of recording')
        # save binary file (chunk by hcunk) into a new file
        raw_filename = output_folder / 'raw_signals.raw'
        #~ recording.write_to_binary_dat_format(raw_filename, time_axis=0, dtype='float32', chunk_mb=500)
        dtype = recording.get_dtype().str
        BinaryRecordingExtractor.write_recording(recording, files_path=[raw_filename],
                                                                time_axis=0, dtype=dtype, chunk_mb=500, verbose=False)
        offset = 0

        # initialize source and probe file
        tdc_dataio = tdc.DataIO(dirname=str(output_folder))
        nb_chan = recording.get_num_channels()

        tdc_dataio.set_data_source(type='RawData', filenames=[str(raw_filename)],
                                   dtype=dtype, sample_rate=recording.get_sampling_frequency(),
                                   total_channel=nb_chan, offset=offset)
        tdc_dataio.set_probe_file(str(prb_file))
        if verbose:
            print(tdc_dataio)

    @classmethod
    def _run_from_folder(cls, output_folder, params, verbose):

        tdc_dataio = tdc.DataIO(dirname=str(output_folder))

        #~ params = dict(self.params)

        clean_catalogue_gui = params.pop('clean_catalogue_gui')
        # make catalogue
        chan_grps = list(tdc_dataio.channel_groups.keys())
        for chan_grp in chan_grps:

            # parameters can change depending the group
            catalogue_nested_params = make_nested_tdc_params(tdc_dataio, chan_grp, **params)

            if verbose:
                print('catalogue_nested_params')
                pprint(catalogue_nested_params)
            
            peeler_params = tdc.get_auto_params_for_peelers(tdc_dataio, chan_grp)
            if verbose:
                print('peeler_params')
                pprint(peeler_params)

            cc = tdc.CatalogueConstructor(dataio=tdc_dataio, chan_grp=chan_grp)
            tdc.apply_all_catalogue_steps(cc, catalogue_nested_params, verbose=verbose)

            if clean_catalogue_gui:
                import pyqtgraph as pg
                app = pg.mkQApp()
                win = tdc.CatalogueWindow(cc)
                win.show()
                app.exec_()

            if verbose:
                print(cc)
            
            #~ if distutils.version.LooseVersion(tdc.__version__) < '1.6.0':
                #~ print('You should upgrade tridesclous')
                #~ t0 = time.perf_counter()
                #~ cc.make_catalogue_for_peeler()
                #~ if self.verbose:
                    #~ t1 = time.perf_counter()
                    #~ print('make_catalogue_for_peeler', t1-t0)

            # apply Peeler (template matching)
            initial_catalogue = tdc_dataio.load_catalogue(chan_grp=chan_grp)
            peeler = tdc.Peeler(tdc_dataio)
            peeler.change_params(catalogue=initial_catalogue, **peeler_params)
            t0 = time.perf_counter()
            peeler.run(duration=None, progressbar=False)
            if verbose:
                t1 = time.perf_counter()
                print('peeler.tun', t1-t0)


    @classmethod
    def _get_result_from_folder(cls, output_folder):
        sorting = TridesclousSortingExtractor(folder_path=output_folder)
        return sorting


def make_nested_tdc_params(tdc_dataio, chan_grp,
                           freq_min=400.,
                           freq_max=5000.,
                           detect_sign='-',
                           detect_threshold=5,
                           peak_span_ms=0.7,
                           wf_left_ms=-2.0,
                           wf_right_ms=3.0,
                           feature_method='auto',
                           cluster_method='auto'):
    params = tdc.get_auto_params_for_catalogue(tdc_dataio, chan_grp=chan_grp)

    params['preprocessor']['highpass_freq'] = freq_min
    params['preprocessor']['lowpass_freq'] = freq_max

    if detect_sign == -1:
        params['peak_detector']['peak_sign'] = '-'
    elif detect_sign == 1:
        params['peak_detector']['peak_sign'] = '+'

    params['peak_detector']['relative_threshold'] = detect_threshold
    params['peak_detector']['peak_span_ms'] = peak_span_ms

    params['extract_waveforms']['wf_left_ms'] = wf_left_ms
    params['extract_waveforms']['wf_right_ms'] = wf_right_ms

    if feature_method != 'auto':
        params['feature_method'] = feature_method
        params['feature_kargs'] = {}

    if cluster_method != 'auto':
        params['cluster_method'] = cluster_method
        params['cluster_kargs'] = {}

    return params
