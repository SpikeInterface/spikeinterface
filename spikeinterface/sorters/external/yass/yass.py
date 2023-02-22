from pathlib import Path
import os
import numpy as np
import sys

from ..basesorter import BaseSorter, get_job_kwargs
from ..utils import ShellScript

from spikeinterface.core import load_extractor

from spikeinterface.core import write_binary_recording
from spikeinterface.extractors import YassSortingExtractor


class YassSorter(BaseSorter):
    """YASS Sorter object."""

    sorter_name = 'yass'
    requires_locations = False
    gpu_capability = 'nvidia-required'
    requires_binary_data = True

    # #################################################

    _default_params = {
        'dtype': 'int16',  # the only datatype that Yass currently accepts;

        # Filtering and processing params
        'freq_min': 300,  # "High-pass filter cutoff frequency",
        'freq_max': 0.3,  # "Low-pass filter cutoff frequency as proportion of sampling rate",
        'neural_nets_path': None,  # default NNs are set to None - Yass will always retrain on dataset;
        'multi_processing': 1,  # 0: single core; 1: multi CPU core
        'n_processors': 1,  # default is a single core; autosearch for more cores
        'n_gpu_processors': 1,  # default is the first installed GPU
        'n_sec_chunk': 10,  # Length of processing chunk in seconds for multi-processing stages
        'n_sec_chunk_gpu_detect': 0.5,  # n_sec_chunk for gpu detection (lower if you get memory error during detection)
        'n_sec_chunk_gpu_deconv': 5,  # n_sec_chunk for gpu deconvolution (lower if you get memory error during deconv)
        'gpu_id': 0,  # which gpu to use, default is 0, i.e. first gpu;
        'generate_phy': 0,  # generate phy visualization files; 0 - do not run; 1: generate phy files
        'phy_percent_spikes': 0.05,
        # generate phy visualization files; ratio of spikes that are processed for phy visualization
        # decrease if memory issues are present

        # params related to NN and clustering;
        'spatial_radius': 70,  # channels spatial radius to consider them neighbors, see
        # yass.geometry.find_channel_neighbors for details

        'spike_size_ms': 5,  # temporal length of templates in ms. It must capture
        # the full shape of waveforms on all channels
        # (reminder: there is a propagation delay in waveform shape across channels)
        # but longer means slower
        'clustering_chunk': [0, 300],  # time (in sec) to run clustering and get initial templates
        # leave blank to run clustering step on entire recording;
        # deconv is then run on the entire dataset using clustering stage templates

        # Params for deconv stage
        'update_templates': 0,  # update templates during deconvolution step
        'neuron_discover': 0,  # recluster during deconvolution and search for new stable neurons;
        'template_update_time': 300,  # if templates being updated, time (in sec) of segment in which to search for
        # new clusters
    }

    _params_description = {

        'dtype': 'int16 : the only datatype that Yass currently accepts',

        # Filtering and processing params
        'freq_min': "300; High-pass filter cutoff frequency",
        'freq_max': "0.3; Low-pass filter cutoff frequency as proportion of sampling rate",
        'neural_nets_path': ' None;  default NNs are set to None - Yass will always retrain on dataset',
        'multi_processing': '1; 0: single core; 1: multi CPU core',
        'n_processors': ' 1; default is a single core; TODO: auto-detect # of corse on node',
        'n_gpu_processors': '1: default is the first installed GPU',
        'n_sec_chunk': '10;  Length of processing chunk in seconds for multi-processing stages. Lower this if running out of memory',
        'n_sec_chunk_gpu_detect': '0.5; n_sec_chunk for gpu detection (lower if you get memory error during detection)',
        'n_sec_chunk_gpu_deconv': '5; n_sec_chunk for gpu deconvolution (lower if you get memory error during deconv)',
        'gpu_id': '0; which gpu ID to use, default is 0, i.e. first gpu',
        'generate_phy': '1; generate phy visualization files; 0 - do not run; 1: generate phy files',
        'phy_percent_spikes': '0.05;  ratio of spikes that are processed for phy visualization; decrease if memory issues are present',

        # params related to NN and clustering;
        'spatial_radius': '70; spatial radius to consider 2 channels neighbors; required for NN stages to work',
        'spike_size_ms': '5; temporal length of templates in ms; longer is more processing time, but slight more accurate',
        # but longer means slower
        'clustering_chunk': '[0, 300]; period of time (in sec) to run clustering and get initial templates; leave blank to run clustering step on entire recording;',

        # Params for deconv stage
        'update_templates': '0; update templates during deconvolution step 1; do not update 0',
        'neuron_discover': '0, recluster during deconvolution and search for new stable neurons: 1; do not recluster 0',
        'template_update_time': '300; if reculstiner on, time (in sec) of segment in which to search for new clusters ',
    }

    # #################################################

    sorter_description = """Yass is a deconvolution and neural network based spike sorting algorithm designed for
                            recordings with no drift (such as retinal recordings).

                            For more information see https://www.biorxiv.org/content/10.1101/2020.03.18.997924v1"""

    installation_mesg = """\nTo install Yass run:\n
                            pip install yass-algorithm

                            Yass can be run in 2 modes:

                            1.  Retraining Neural Networks (Default)

                            rec, sort = se.toy_example(duration=300)
                            sorting_yass = ss.run_yass(rec, '/home/cat/Downloads/test2')


                            2.  Using previously trained Neural Networks:
                            ...
                            sorting_yass = ss.run_yass(rec, '/home/cat/Downloads/test2', neural_nets_path=PATH_TO_NEURAL_NETS)

                            For any installation or operation issues please visit: https://github.com/paninski-lab/yass

                        """

    @classmethod
    def is_installed(cls):
        try:
            import yaml
            import yass
            HAVE_YASS = True
        except ImportError:
            HAVE_YASS = False
        return HAVE_YASS

    @classmethod
    def get_sorter_version(cls):
        import yass
        return yass.__version__

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        import yaml

        p = params

        source_dir = Path(__file__).parent
        config_default_location = os.path.join(source_dir, 'config_default.yaml')

        with open(config_default_location) as file:
            yass_params = yaml.load(file, Loader=yaml.FullLoader)

            # update root folder
        yass_params['data']['root_folder'] = str(sorter_output_folder.absolute())

        #  geometry
        probe_file_txt = os.path.join(sorter_output_folder, 'geom.txt')
        geom_txt = recording.get_channel_locations()
        np.savetxt(probe_file_txt, geom_txt)

        #   params
        yass_params['recordings']['sampling_rate'] = recording.get_sampling_frequency()
        yass_params['recordings']['n_channels'] = recording.get_num_channels()

        # save to int16 raw
        input_file_path = os.path.join(sorter_output_folder, 'data.bin')
        dtype = 'int16'  # HARD CODE THIS FOR YASS
        input_file_path = sorter_output_folder / 'data.bin'
        
        write_binary_recording(recording, file_paths=[input_file_path], dtype=dtype, **get_job_kwargs(params, verbose))

        retrain = False
        if params['neural_nets_path'] is None:
            params['neural_nets_path'] = str(sorter_output_folder / 'tmp' / 'nn_train')
            retrain = True

        # MERGE yass_params with self.params that could be changed by the user
        merge_params = merge_params_dict(yass_params, params)

        # to yaml
        fname_config = sorter_output_folder / 'config.yaml'
        with open(fname_config, 'w') as file:
            documents = yaml.dump(merge_params, file)

        # RunNN training on existing
        neural_nets_path = p['neural_nets_path']

        if retrain:
            # retrain NNs
            YassSorter.train(recording, sorter_output_folder, verbose)

            # update NN folder location
            neural_nets_path = sorter_output_folder / 'tmp' / 'nn_train'
        else:
            #   load previous NNs
            if verbose:
                print("USING PREVIOUSLY TRAINED NNs FROM THIS LOCATION: ", params['neural_nets_path'])
            # use previously trained NN folder location
            neural_nets_path = Path(params['neural_nets_path'])

        merge_params['neuralnetwork']['denoise']['filename'] = str(neural_nets_path.absolute() / 'denoise.pt')
        merge_params['neuralnetwork']['detect']['filename'] = str(neural_nets_path.absolute() / 'detect.pt')

        # to yaml again (for NNs update)
        fname_config = sorter_output_folder / 'config.yaml'
        with open(fname_config, 'w') as file:
            yaml.dump(merge_params, file)

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        '''
        '''
        config_file = sorter_output_folder.absolute() / 'config.yaml'
        if 'win' in sys.platform and sys.platform != 'darwin':
            shell_cmd = f'''yass sort {config_file}'''
        else:
            shell_cmd = f'''
                        #!/bin/bash
                        yass sort {config_file}'''

        shell_script = ShellScript(shell_cmd,
                                   #  script_path=os.path.join(sorter_output_folder, self.sorter_name),
                                   script_path=sorter_output_folder / 'run_yass',
                                   log_path=sorter_output_folder / (cls.sorter_name + '.log'),
                                   verbose=verbose)
        shell_script.start()

        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception('yass returned a non-zero exit code')

    # Alessio might not want to put here;
    # better option to have a parameter "tune_nn" which
    @classmethod
    def train(cls, recording, sorter_output_folder, verbose):
        ''' Train NNs on yass prior to running yass sort'''

        if verbose:
            print(
                "TRAINING YASS (Note: using default spike width, neighbour chan radius; to change, see parameter files)")
            print("To use previously-trained NNs, change the NNs prior to running: ")
            print("            ss.set_NNs('path_to_NNs') (or set params['neural_nets_path'] = path_toNNs)")
            print("prior to running ss.run_sorter()")

        config_file = sorter_output_folder.absolute() / 'config.yaml'
        if 'win' in sys.platform and sys.platform != 'darwin':
            shell_cmd = f'yass train {config_file}'
        else:
            shell_cmd = f'''
                        #!/bin/bash
                        yass train {config_file}'''

        shell_script = ShellScript(shell_cmd,
                                   script_path=sorter_output_folder / 'run_yass_train',
                                   # os.path.join(sorter_output_folder, cls.sorter_name),
                                   log_path=sorter_output_folder / (cls.sorter_name + '_train.log'),
                                   verbose=verbose)
        shell_script.start()

        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception('yass returned a non-zero exit code')

        if verbose:
            print("TRAINING COMPLETED. NNs located at: ", sorter_output_folder,
                  "/tmp/nn_train/detect.pt and ",
                  sorter_output_folder, "/tmp/nn_train/denoise.pt")

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        sorting = YassSortingExtractor(folder_path=Path(sorter_output_folder))
        return sorting

    # TODO integrate this logic somewhere or remove ????
    # def neural_nets_default(self, sorter_output_folder):
    # ''' Revert to default NNs
    # '''
    # self.merge_params['neuralnetwork']['denoise']['filename'] = 'denoise.pt'
    # self.merge_params['neuralnetwork']['detect']['filename'] = 'detect.pt'
    # fname_config = os.path.join(sorter_output_folder, 'config.yaml')
    # with open(fname_config, 'w') as file:
    # documents = yaml.dump(self.merge_params, file)


def merge_params_dict(yass_params, params):
    ''' This function merges params with yass_params (default)
        to make a larger exposed params dictionary
    '''
    # self.params
    # self.yass_params

    merge_params = yass_params.copy()

    merge_params['preprocess']['filter']['low_pass_freq'] = params['freq_min']
    merge_params['preprocess']['filter']['high_factor'] = params['freq_max']

    merge_params['neuralnetwork']['detect']['filename'] = os.path.join(params['neural_nets_path'], 'detect.pt')
    merge_params['neuralnetwork']['denoise']['filename'] = os.path.join(params['neural_nets_path'], 'denoise.pt')

    merge_params['resources']['multi_processing'] = params['multi_processing']
    merge_params['resources']['n_processors'] = params['n_processors']
    merge_params['resources']['n_gpu_processors'] = params['n_gpu_processors']
    merge_params['resources']['n_sec_chunk'] = params['n_sec_chunk']
    merge_params['resources']['n_sec_chunk_gpu_detect'] = params['n_sec_chunk_gpu_detect']
    merge_params['resources']['n_sec_chunk_gpu_deconv'] = params['n_sec_chunk_gpu_deconv']
    merge_params['resources']['gpu_id'] = params['gpu_id']
    merge_params['resources']['generate_phy'] = params['generate_phy']
    merge_params['resources']['phy_percent_spikes'] = params['phy_percent_spikes']

    merge_params['recordings']['spatial_radius'] = params['spatial_radius']
    merge_params['recordings']['spike_size_ms'] = params['spike_size_ms']
    merge_params['recordings']['clustering_chunk'] = params['clustering_chunk']

    merge_params['deconvolution']['update_templates'] = params['update_templates']
    merge_params['deconvolution']['neuron_discover'] = params['neuron_discover']
    merge_params['deconvolution']['template_update_time'] = params['template_update_time']

    return merge_params
