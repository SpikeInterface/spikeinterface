import numpy as np
import scipy.io


def generate_ops_file(options, output_folder):
    """
    This function generates kilosort3 ops (configs) and saves as `ops.mat`

    Based on kilosort3_config.m file, creates a .mat file to be loaded in kilosort3_main.m

    Loading example in Matlab (should be assigned to a variable called `ops`):
    >> ops = load('/output_folder/ops.mat');

    Args:
        options (dict): Dict with user parameters
        output_folder (pathlib.Path): Path to save `ops.mat`
    """

    # Converting integer values into float
    # Kilosort3 interprets ops fields as double
    for k, v in options.items():
        if isinstance(v, int):
            options[k] = float(v)
    options['projection_threshold'] = [float(pt) for pt in options['projection_threshold']]

    ops = dict()

    ops['NchanTOT'] = options['nchan']  # total number of channels (omit if already in chanMap file)
    ops['Nchan'] = options['nchan']  # number of active channels (omit if already in chanMap file)

    ops['datatype'] = 'dat'  # binary ('dat', 'bin') or 'openEphys'
    ops['fbinary'] = options['dat_file']  # will be created for 'openEphys'
    ops['fproc'] = options['temp_wh_file']  # residual from RAM of preprocessed data
    ops['root'] = options['root']  # 'openEphys' only: where raw files are
    ops['trange'] = [0, np.Inf] #  time range to sort

    # define the channel map as a filename (string) or simply an array
    ops['chanMap'] = options['chan_map']

    # sample rate
    ops['fs'] = options['sample_rate']

    # frequency for high pass filtering (150)
    ops['fshigh'] = options['freq_min']

    # threshold on projections (like in Kilosort1, can be different for last pass like [10 4])
    ops['Th'] = options['projection_threshold']

    # how important is the amplitude penalty (like in Kilosort1, 0 means not used, 10 is average, 50 is a lot)
    ops['lam'] = 20.0

    # splitting a cluster at the end requires at least this much isolation for each sub-cluster (max = 1)
    ops['AUCsplit'] = 0.8

    # minimum firing rate on a "good" channel (0 to skip)
    ops['minfr_goodchannels'] = options['minfr_goodchannels']

    # minimum spike rate (Hz), if a cluster falls below this for too long it gets removed
    ops['minFR'] = options['minFR']

    # spatial constant in um for computing residual variance of spike
    ops['sigmaMask'] = options['sigmaMask']

    # threshold crossings for pre-clustering (in PCA projection space)
    ops['ThPre'] = options['preclust_threshold']

    # spatial scale for datashift kernel
    ops['sig'] = options['sig']

    # type of data shifting (0 = none, 1 = rigid, 2 = nonrigid)
    ops['nblocks'] = options['nblocks']

    ops['CAR'] = options['use_car']

    ## danger, changing these settings can lead to fatal errors
    # options for determining PCs
    ops['spkTh'] = -options['detect_threshold']  # spike threshold in standard deviations (-6)
    ops['reorder'] = 1.0  # whether to reorder batches for drift correction. 
    ops['nskip'] = 25.0  # how many batches to skip for determining spike PCs

    ops['GPU'] = 1.0  # has to be 1, no CPU version yet, sorry
    # ops['Nfilt'] = 1024 # max number of clusters
    ops['nfilt_factor'] = options['nfilt_factor']  # max number of clusters per good channel (even temporary ones)
    ops['ntbuff'] = options['ntbuff']  # samples of symmetrical buffer for whitening and spike detection
    ops['NT'] = options['NT']  # must be multiple of 32 + ntbuff. This is the batch size (try decreasing if out of memory). 
    ops['whiteningRange'] = 32.0  # number of channels to use for whitening each channel
    ops['nSkipCov'] = 25.0  # compute whitening matrix from every N-th batch
    ops['scaleproc'] = 200.0  # int16 scaling of whitened data
    ops['nPCs'] = options['nPCs']  # how many PCs to project the spikes into
    ops['useRAM'] = 0.0  # not yet available

    scipy.io.savemat(str(output_folder / 'ops.mat'), ops)
