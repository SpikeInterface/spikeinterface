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
from .utils import SpikeSortingError




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
                remove_existing_folder=False, delete_output_folder=False, ):
        
        output_folder = self.initialize_folder(recording, output_folder, verbose, remove_existing_folder)
        
        self.recording = recording
        self.verbose = verbose
        self.delete_output_folder = delete_output_folder
        self.output_folder = output_folder
    
    def set_params(self, sorter_params):
        """
        Mimic the old API
        This should not be used anymore but still works.
        """
        p = self.set_params_to_folder(self.recording, self.output_folder, sorter_params, self.verbose)
        self.params = p
    
    def run(self, raise_error=True):
        """
        Main function keept for backward compatibility.
        This should not be used anymore but still works.
        """
        # setup recording
        self.setup_recording(self.recording, self.output_folder, self.params, self.verbose)
        
        # compute
        self.run_from_folder(self.output_folder, raise_error=True)

 
    def get_result(self):
        sorting = self.get_result_from_folder(self.output_folder)
        if self.delete_output_folder:
            shutil.rmtree(str(self.output_folder), ignore_errors=True)
        return sorting
    
    
    #############################################
    
    # class method zone
    
    @classmethod
    def initialize_folder(cls, recording, output_folder, verbose, remove_existing_folder):
        assert cls.is_installed(), """The sorter {} is not installed.
        Please install it with:  \n{} """.format(cls.sorter_name, cls.installation_mesg)
        
        if cls.requires_locations:
            locations = recording.get_channel_locations()
            if locations is None:
                raise RuntimeError("Channel locations are required for this spike sorter. "
                                   "Locations can be added to the RecordingExtractor by loading a probe file "
                                   "(.prb or .csv) or by setting them manually.")

        params = cls.default_params()

        if output_folder is None:
            output_folder = cls.sorter_name + '_output'
        
        # .absolute() not anymore
        output_folder = Path(output_folder) 

        print('remove_existing_folder', remove_existing_folder)
        if output_folder.is_dir():
            if remove_existing_folder:
                shutil.rmtree(str(output_folder))
            else:
                raise ValueError(f'Folder {output_folder} already exsits')
        
    
        output_folder.mkdir(parents=True, exist_ok=True)

        
        if recording.get_num_segments() > 1:
            if not cls.handle_multi_segment:
                raise ValueError(f'This sorter {cls.sorter_name} do not handle multi segment, use recording.split(by=...)')
        
        #~ self.recording = recording
        
        rec_file = output_folder / 'spikeinterface_recording.json'
        #~ assert recording.is_dumpable
        #~ recording.dump_to_json(rec_file)
        if recording.is_dumpable:
            recording.dump_to_json(rec_file)
        else:
            d = {'warning': 'The recording is not dumpable'}
            rec_file.write_text(json.dumps(d, indent=4), encoding='utf8')

        return output_folder
    
    @classmethod
    def default_params(cls):
        return copy.deepcopy(cls._default_params)

    @classmethod
    def params_description(cls):
        return copy.deepcopy(cls._params_description)
    
    @classmethod
    def set_params_to_folder(cls, recording, output_folder, new_params, verbose):
        params = cls.default_params()

        # verify paarms are in list
        bad_params = []
        for p in new_params.keys():
            if p not in params.keys():
                bad_params.append(p)
        if len(bad_params) > 0:
            raise AttributeError('Bad parameters: ' + str(bad_params))
        
        params.update(new_params)
        
        # custum check params
        params = cls._check_params(recording, output_folder, params)
        # common check : filter warning
        if recording.is_filtered and cls._check_already_filtered(params) and verbose:
            print(f"Warning! The recording is already filtered, but {cls.sorter_name} filter is enabled")

        # dump parameters inside the folder with json
        cls._dump_params(recording, output_folder, params, verbose)
        
        return params
        
    @classmethod
    def _dump_params(cls, recording, output_folder, sorter_params, verbose):
        with (output_folder / 'spikeinterface_params.json').open(mode='w', encoding='utf8') as f:
            params = dict()
            params['sorter_name'] = cls.sorter_name
            params['sorter_params'] = sorter_params
            params['verbose'] = verbose
            json.dump(check_json(params), f, indent=4)

    
    @classmethod
    def setup_recording(cls, recording, output_folder, params, verbose):
        output_folder = Path(output_folder)
        cls._setup_recording(recording, output_folder, params, verbose)
        
    @classmethod
    def run_from_folder(cls, output_folder, raise_error):
        # need setup_recording to be done.
        output_folder = Path(output_folder)
        
        # retrieve sorter_name and params
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
        
        #~ SorterClass._run_from_folder(output_folder, sorter_params, verbose)
        
        try:
            SorterClass._run_from_folder(output_folder, sorter_params, verbose)
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
            raise SpikeSortingError("Spike sorting failed. You can inspect the runtime trace in spikeinterface_log.json")
        
        return run_time

    @classmethod
    def get_result_from_folder(cls, output_folder):
        output_folder = Path(output_folder)
        # check erros in log file
        log_file  = output_folder / 'spikeinterface_log.json'
        if not log_file.is_file():
            raise SpikeSortingError('get reult error: the folder do not contain spikeinterface_log.json')

        with log_file.open('r', encoding='utf8') as f:
            log = json.load(f)
        
        if bool(log['error']):
            raise SpikeSortingError("Spike sorting failed. You can inspect the runtime trace in spikeinterface_log.json")
        
        sorting = cls._get_result_from_folder(output_folder)
        return sorting

    #############################################
    
    # Zone to be implemeneted
    # by design all are implemented with class method.
    # No instance!!
    # So "self" is not available. Everything is folder based.
    # This should help for computing distribution

    @classmethod
    def get_sorter_version(cls):
        # need be implemented in subclass
        raise NotImplementedError

    @classmethod
    def _check_params(cls, recording, output_folder, params):
        # need be implemented in subclass for custum checks
        # here nothing to change
        return params

    @classmethod
    def _setup_recording(cls, recording, output_folder, params, verbose):
        # need be implemented in subclass
        # this setup ONE recording (or SubExtractor)
        # this must copy (or not) the trace in the appropirate format
        # this must take care of geometry file (PRB, CSV, ...)
        # this must generate all needed script
        raise NotImplementedError

    @classmethod
    def is_installed(cls):
        # need be implemented in subclass
        raise NotImplementedError

    @classmethod
    def _check_already_filtered(cls, params):
        return False
        # need be implemented in subclass for custum checks

    @classmethod
    def _run_from_folder(cls, output_folder, params, verbose):
        # need be implemented in subclass
        # this is where the script is launch for one recording from a folder already prepared
        # this must run or generate the command line to run the sorter for one recording
        raise NotImplementedError

    @classmethod
    def _get_result_from_folder(cls, output_folder):
        # need be implemented in subclass
        raise NotImplementedError
    
    
    


