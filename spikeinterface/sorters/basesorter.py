"""
base class for sorters implementation.
"""
import time
import copy
from pathlib import Path
import os
import datetime
import json
import traceback
import shutil

import numpy as np

from joblib import Parallel, delayed

from spikeinterface.core import load_extractor
from spikeinterface.core.core_tools import check_json
from .sorter_tools import SpikeSortingError




class BaseSorter:
    sorter_name = ''  # convinience for reporting
    SortingExtractor_Class = None  # convinience to get the extractor
    requires_locations = False
    compatible_with_parallel = {'loky': True, 'multiprocessing': True, 'threading': True}
    _default_params = {}
    _params_description = {}
    sorter_description = ""
    installation_mesg = ""  # error message when not installed
    
    # by default no sorters handle multi segment
    handle_multi_segment = False

    def __init__(self, recording=None, output_folder=None, verbose=False,
                 delete_output_folder=False):
                     #~ grouping_property=None, 

        assert self.is_installed(), """The sorter {} is not installed.
        Please install it with:  \n{} """.format(self.sorter_name, self.installation_mesg)
        if self.requires_locations:
            locations = recording.get_channel_locations()
            if locations is None:
                raise RuntimeError("Channel locations are required for this spike sorter. "
                                   "Locations can be added to the RecordingExtractor by loading a probe file "
                                   "(.prb or .csv) or by setting them manually.")

        self.verbose = verbose
        #~ self.grouping_property = grouping_property
        self.params = self.default_params()

        if output_folder is None:
            output_folder = self.sorter_name + '_output'
        output_folder = Path(output_folder).absolute()
        

        if output_folder.is_dir():
            shutil.rmtree(str(output_folder))

        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)

        
        if recording.get_num_segments() > 1:
            if not self.handle_multi_segment:
                raise ValueError(f'This sorter {self.sorter_name} do not handle multi segment, use recording.split(by=...)')
        
        self.recording = recording
        
        rec_file = self.output_folder / 'spikeinterface_recording.json'
        if recording.is_dumpable:
            recording.dump_to_json(rec_file)
        else:
            d = {'warning': 'The recording is not dumpable'}
            rec_file.write_text(json.dumps(d, indent=4), encoding='utf8')
        
        self.delete_output_folder = delete_output_folder

    @classmethod
    def default_params(cls):
        return copy.deepcopy(cls._default_params)

    @classmethod
    def params_description(cls):
        return copy.deepcopy(cls._params_description)
        
    def set_params(self, **params):
        bad_params = []
        for p in params.keys():
            if p not in self._default_params.keys():
                bad_params.append(p)
        if len(bad_params) > 0:
            raise AttributeError('Bad parameters: ' + str(bad_params))
        self.params.update(params)

        # dump parameters inside the folder with json
        self._dump_params()
        
        # filter warning
        if self.recording.is_filtered and self._check_already_filtered(params) and self.verbose:
            print(f"Warning! The recording is already filtered, but {self.sorter_name} filter is enabled")

    def _dump_params(self):
        #~ for output_folder, recording in zip(self.output_folders, self.recording_list):
            #~ with open(str(output_folder / 'spikeinterface_params.json'), 'w', encoding='utf8') as f:
                #~ params = dict()
                #~ params['sorter_params'] = self.params
                #~ # only few properties are put to json
                #~ params['recording'] = recording.to_dict(include_properties=False, include_features=False)
                #~ json.dump(check_json(params), f, indent=4)

        with (self.output_folder / 'spikeinterface_params.json').open(mode='w', encoding='utf8') as f:
            params = dict()
            params['sorter_name'] = self.sorter_name
            params['sorter_params'] = self.params
            params['verbose'] = self.verbose
            
            # only few properties/features are put to json
            #~ params['recording'] = self.recording.to_dict(include_properties=False, include_features=False)
            #~ from pprint import pprint
            #~ pprint(params)
            json.dump(check_json(params), f, indent=4)

    def run(self, raise_error=True):
        """
        Main function keept for backward compatibility.
        This should not be used anymore.
        """
        # setup recording
        self._setup_recording(self.recording, self.output_folder)
        
        # compute
        self.compute_from_folder(self.output_folder, raise_error=True)

    @staticmethod
    def compute_from_folder(output_folder, raise_error=True):
        # need setup_recording to be done.
        output_folder = Path(output_folder)
        
        # retrieve sorter and params
        with (output_folder / 'spikeinterface_params.json').open(mode='r') as f:
            params = json.load(f)
        sorter_params = params['sorter_params']
        sorter_name = params['sorter_name']
        verbose = params['verbose']

        from .sorterlist import sorter_dict
        SorterClass = sorter_dict[sorter_name]
        
        # not needed normally
        # recording = load_extractor(output_folder / 'spikeinterface_recording.json')

        now = datetime.datetime.now()
        log = {
            'sorter_name': str(SorterClass.sorter_name),
            'sorter_version': str(SorterClass.get_sorter_version()),
            'datetime': now,
            'runtime_trace': []
        }
        t0 = time.perf_counter()
        
        SorterClass._compute_from_folder(output_folder, sorter_params, verbose)
        
        try:
            SorterClass._compute_from_folder(output_folder, sorter_params, verbose)
            t1 = time.perf_counter()
            run_time = float(t1 - t0)
            has_error = False
        except Exception as err:
            has_error = True
            run_time = None
            log['error'] = True
            log['error_trace'] = traceback.format_exc()
        
        log['error'] = has_error
        log['run_time'] = run_time

        # some sorter have a log file dur to shellscript launcher
        runtime_trace_path = output_folder / f'{sorter_name}.log'
        runtime_trace = []
        if runtime_trace_path.is_file():
            with open(runtime_trace_path, 'r') as fp:
                line = fp.readline()
                while line:
                    runtime_trace.append(line.strip())
                    line = fp.readline()
        log['runtime_trace'] = runtime_trace
        
        # dump to json
        with (output_folder / 'spikeinterface_log.json').open('w', encoding='utf8') as f:
            json.dump(check_json(log), f, indent=4)

        if verbose:
            if has_error:
                print(f'Error running {sorter_name}')
            else:
                print(f'{sorter_name} run time {run_time:0.2f}s')

        if has_error and raise_error:
            raise SpikeSortingError(f"Spike sorting failed. You can inspect the runtime trace in "
                                    f"the {sorter_name}.log of the output folder.'")
        
        return run_time

    def get_result(self):
        sorting = self.get_result_from_folder(self.output_folder)

        if self.delete_output_folder:
            #~ for out in self.output_folders:
                #~ if self.verbose:
                    #~ print("Removing ", str(out))
                #~ shutil.rmtree(str(out), ignore_errors=True)
            shutil.rmtree(str(self.output_folder), ignore_errors=True)
        
        if sorting.get_sampling_frequency() is None:
            sorting.set_sampling_frequency(self.recording.get_sampling_frequency())

        return sorting

    #############################################"
    
    # Zone to be implemeneted
    def _setup_recording(self, recording, output_folder):
        # need be implemented in subclass
        # this setup ONE recording (or SubExtractor)
        # this must copy (or not) the trace in the appropirate format
        # this must take care of geometry file (PRB, CSV, ...)
        # this must generate all needed script
        raise NotImplementedError
    
    @classmethod
    def get_sorter_version(cls):
        # need be implemented in subclass
        raise NotImplementedError
    
    @classmethod
    def is_installed(cls):
        # need be implemented in subclass
        raise NotImplementedError
    
    @classmethod
    def _check_already_filtered(cls, params):
        return False
        # need be implemented in subclass for custum checks
    

    #~ def _run(self, recording, output_folder):
        #~ # need be implemented in subclass
        #~ # this run the sorter on ONE recording (or SubExtractor)
        #~ # this must run or generate the command line to run the sorter for one recording
        #~ raise NotImplementedError
    
    @classmethod
    def _compute_from_folder(cls, output_folder, params, verbose):
        # need be implemented in subclass
        # this is where the script is launch for one recording from a folder already prepared
        # this must run or generate the command line to run the sorter for one recording
        raise NotImplementedError

    @classmethod
    def get_result_from_folder(cls, output_folder):
        # need be implemented in subclass
        raise NotImplementedError
    
    
    


