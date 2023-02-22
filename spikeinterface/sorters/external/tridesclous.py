from pathlib import Path
import os
import shutil
import numpy as np
import copy
import time
from pprint import pprint

from spikeinterface.extractors import TridesclousSortingExtractor

from ..basesorter import BaseSorter, get_job_kwargs
from spikeinterface.core import write_binary_recording

from probeinterface import write_prb


class TridesclousSorter(BaseSorter):
    """Tridesclous Sorter object."""

    sorter_name = 'tridesclous'
    requires_locations = False
    compatible_with_parallel = {'loky': True, 'multiprocessing': False, 'threading': False}
    requires_binary_data = True

    _default_params = {
        'freq_min': 400.,
        'freq_max': 5000.,
        'detect_sign': -1,
        'detect_threshold': 5,
        'common_ref_removal': False,
        'nested_params': None,
    }

    _params_description = {
        'freq_min': "High-pass filter cutoff frequency",
        'freq_max': "Low-pass filter cutoff frequency",
        'detect_threshold': "Threshold for spike detection",
        'detect_sign': "Use -1 (negative) or 1 (positive) depending "
                       "on the sign of the spikes in the recording",
        'common_ref_removal': 'remove common reference with median',
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
    handle_multi_segment = True

    @classmethod
    def is_installed(cls):
        try:
            import tridesclous as tdc
            HAVE_TDC = True
        except ImportError:
            HAVE_TDC = False
        except:
            print('tridesclous is installed, but it has some dependency problems, check numba or hdbscan installations!')
            HAVE_TDC = False
        return HAVE_TDC

    @classmethod
    def get_sorter_version(cls):
        import tridesclous as tdc
        return tdc.__version__

    @classmethod
    def _check_params(cls, recording, sorter_output_folder, params):
        return params

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        import tridesclous as tdc

        # save prb file
        probegroup = recording.get_probegroup()
        prb_file = sorter_output_folder / 'probe.prb'
        write_prb(prb_file, probegroup)

        num_seg = recording.get_num_segments()
        sr = recording.get_sampling_frequency()

        if recording.binary_compatible_with(time_axis=0):
            # no copy
            d = recording.get_binary_description()
            file_paths = d['file_paths']
            dtype = str(d['dtype'])
            num_chan = d['num_channels']
            file_offset = d['file_offset']
        else:
            if verbose:
                print('Local copy of recording')
            # save binary file (chunk by chunk) into a new file
            num_chan = recording.get_num_channels()
            dtype = recording.get_dtype().str
            file_paths = [str(sorter_output_folder / f'raw_signals_{i}.raw') for i in range(num_seg)]
            write_binary_recording(recording, file_paths=file_paths,
                                   dtype=dtype, verbose=False, **get_job_kwargs(params, verbose))
            file_offset = 0

        # initialize source and probe file
        tdc_dataio = tdc.DataIO(dirname=str(sorter_output_folder))

        tdc_dataio.set_data_source(type='RawData', filenames=file_paths,
                                   dtype=dtype, sample_rate=float(sr),
                                   total_channel=int(num_chan), offset=int(file_offset))
        tdc_dataio.set_probe_file(str(prb_file))
        if verbose:
            print(tdc_dataio)

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        import tridesclous as tdc

        tdc_dataio = tdc.DataIO(dirname=str(sorter_output_folder))

        params = params.copy()

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

            if verbose:
                print(cc)

            # apply Peeler (template matching)
            initial_catalogue = tdc_dataio.load_catalogue(chan_grp=chan_grp)
            peeler = tdc.Peeler(tdc_dataio)
            peeler.change_params(catalogue=initial_catalogue, **peeler_params)
            t0 = time.perf_counter()
            peeler.run(duration=None, progressbar=False)
            if verbose:
                t1 = time.perf_counter()
                print('peeler.tun', t1 - t0)

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        sorting = TridesclousSortingExtractor(folder_path=sorter_output_folder)
        return sorting


def make_nested_tdc_params(tdc_dataio, chan_grp, **new_params):
    import tridesclous as tdc

    params = tdc.get_auto_params_for_catalogue(tdc_dataio, chan_grp=chan_grp)

    if 'freq_min' in new_params:
        params['preprocessor']['highpass_freq'] = new_params['freq_min']

    if 'freq_max' in new_params:
        params['preprocessor']['lowpass_freq'] = new_params['freq_max']

    if 'common_ref_removal' in new_params:
        params['preprocessor']['common_ref_removal'] = new_params['common_ref_removal']

    if 'detect_sign' in new_params:
        detect_sign = new_params['detect_sign']
        if detect_sign == -1:
            params['peak_detector']['peak_sign'] = '-'
        elif detect_sign == 1:
            params['peak_detector']['peak_sign'] = '+'

    if 'detect_threshold' in new_params:
        params['peak_detector']['relative_threshold'] = new_params['detect_threshold']

    nested_params = new_params.get('nested_params', None)
    if nested_params is not None:
        for k, v in nested_params.items():
            if isinstance(v, dict):
                params[k].update(v)
            else:
                params[k] = v

    return params
