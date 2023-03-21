# Standard Libraries
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import copy

# Spikeinterface
import spikeinterface.core as sc
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from spikeinterface.sortingcomponents.matching import find_spikes_from_templates

# SpikePSVAE
from spike_psvae.deconvolve import MatchPursuitObjectiveUpsample as Objective


def test_matching_psvae():
    verbose = False
    down_factor = 5
    hit_threshold = 0.4e-3
    plot_templates = False

    filepaths = generate_filepaths(project_name="test_matching_psvae", verbose=verbose)
    job_kwargs = dict(n_jobs=4, chunk_duration='1s', progress_bar=True, verbose=verbose)
    recording, gt_sorting = generate_gt_recording(job_kwargs, filepaths, verbose=verbose)
    templates, we = generate_templates(recording, gt_sorting, job_kwargs, filepaths, overwrite=False, verbose=verbose)
    recording, gt_sorting = shorten_recording(recording, gt_sorting, filepaths, verbose=verbose)
    downsampled_output = downsample_recording(recording, gt_sorting, templates, we, filepaths,
                                              down_factor=down_factor, overwrite=True, verbose=verbose)
    recording, gt_sorting, gt_templates, nbefore, nafter = downsampled_output
    if verbose:
        print(f"dt = {1 / recording.sampling_frequency}")

    if plot_templates:
        dt = 1/we.sampling_frequency
        t = np.arange(-we.nbefore*dt, we.nafter*dt, dt)
        dt_down = 1/recording.sampling_frequency
        t_down = np.arange(-nbefore*dt_down, nafter*dt_down, dt_down)
        plt.figure()
        best_ch = np.argmax(np.max(np.abs(templates[0]), axis=0))
        plt.plot(t, templates[0, :, best_ch], c='k', label='True Template')
        plt.plot(t_down, gt_templates[0, :, best_ch], c='r', ls='--', label='Downsampled Template')
        plt.show()

    duplicated_templates = np.concatenate((gt_templates, gt_templates[ [0] ]))
    method_kwargs = generate_method_kwargs(recording, nbefore, nafter, verbose=verbose)
    template_ids2unit_ids = list(range(7))
    template_ids2unit_ids.append(0)
    template_ids2unit_ids = np.array(template_ids2unit_ids)
    param_sets = {
        "specify all parameters" : dict(lambd=0, n_jobs=1, template_ids2unit_ids=None, jitter_factor=1, vis_su=1,
                                        threshold=50),
        "check amplitude scaling, multiprocessing, grouped templates" : dict(
            lambd=1, n_jobs=2, template_ids2unit_ids=template_ids2unit_ids, jitter_factor=1, vis_su=1),
        "check trivial cases": dict(lambd=0, n_jobs=1, template_ids2unit_ids=None, jitter_factor=1),
        "check no_amplitude_scaling" : dict(lambd=0, n_jobs=1, template_ids2unit_ids=None, jitter_factor=down_factor),
        "check sparsity": dict(lambd=0, n_jobs=1, template_ids2unit_ids=None, jitter_factor=down_factor, vis_su=10),
        "check best case" : dict(lambd=1, n_jobs=1, template_ids2unit_ids=None, jitter_factor=2*down_factor)
    }
    for params in param_sets.values():
        print(f"{params = }")
        job_kwargs['n_jobs'] = params.pop('n_jobs')
        if params['template_ids2unit_ids'] is None:
            templates = gt_templates
        else:
            templates = duplicated_templates
        method_kwargs['objective_kwargs']['templates'] = templates
        method_kwargs['objective_kwargs'].update(params)
        spikes = run_matching(recording, method_kwargs, job_kwargs, verbose=verbose)
        psvae_kwargs = copy.deepcopy(method_kwargs)
        spikes_psvae = run_spike_psvae(templates, psvae_kwargs, filepaths, verbose=verbose)
        hit_rate, misclass_rate, miss_rate = evaluate_performance(recording, gt_sorting, we, spikes,
                                                                  hit_threshold=hit_threshold, verbose=verbose)
        hit_psvae, misclass_psvae, miss_psvae = evaluate_performance(recording, gt_sorting, we, spikes,
                                                                  hit_threshold=hit_threshold, verbose=verbose)

        print("Performance:")
        print(f"{hit_rate = }")
        print(f"{misclass_rate = }")
        print(f"{miss_rate = }")
        print(f"# of spikes detected = {spikes.shape[0]}")
        print(f"# of true spikes = {len(gt_sorting.get_all_spike_trains()[0][0])}")

        assert hit_rate == hit_psvae, "Hit Rate doesn't match Spike-PSVAE"
        assert misclass_rate == misclass_psvae, "Misclass Rate doesn't match Spike-PSVAE"
        assert miss_rate == miss_psvae, "Miss Rate doesn't match Spike-PSVAE"
        assert len(spikes) == len(spikes_psvae), "# of Detected spikes doesn't match Spike-PSVAE"


def generate_filepaths(project_name, verbose=False):
    if verbose:
        print("...Generating Filepaths...")
    base_path = Path("/Volumes/T7/CatalystNeuro")
    mearec_path = base_path / "MEArecDatasets/recordings/toy_recording_30min.h5"
    project_path = base_path / project_name
    rec_path = project_path / "recording"
    rec_short_path = project_path / "rec_short"
    rec_down_path = project_path / "rec_down"
    sort_down_path = project_path / "sort_down.npz"
    we_path = project_path / "waveforms"
    filepaths = dict(
        base_path = base_path,
        mearec_path = mearec_path,
        project_path = project_path,
        rec_path = rec_path,
        rec_short_path = rec_short_path,
        rec_down_path = rec_down_path,
        sort_down_path = sort_down_path,
        we_path = we_path,
    )
    if not filepaths['project_path'].exists():
        filepaths['project_path'].mkdir()
    return filepaths


def generate_gt_recording(job_kwargs, filepaths, verbose=False):
    if verbose:
        print("...Generating Recording...")
    mearec_path, rec_path = filepaths['mearec_path'], filepaths['rec_path']
    if not rec_path.exists():
        recording, _ = se.read_mearec(mearec_path)
        recording = spre.bandpass_filter(recording, dtype='float32')
        recording = spre.common_reference(recording)
        #recording = spre.whiten(recording) # whitening induces an error for some reason
        recording.save(folder=rec_path, **job_kwargs)
    recording = sc.load_extractor(rec_path)
    _, gt_sorting = se.read_mearec(mearec_path)
    return recording, gt_sorting


def generate_templates(recording, sorting, job_kwargs, filepaths, overwrite=False, verbose=False):
    if verbose:
        print("...Generating Templates...")
    if overwrite:
        we_kwargs = dict(ms_before=2.5, ms_after=3.5, max_spikes_per_unit=500, overwrite=True)
    else:
        we_kwargs = dict(ms_before=2.5, ms_after=3.5, max_spikes_per_unit=500, load_if_exists=True)
    we = sc.extract_waveforms(recording, sorting, filepaths['we_path'], **we_kwargs, **job_kwargs)
    templates = we.get_all_templates(we.unit_ids)
    return templates, we


def generate_method_kwargs(recording, nbefore, nafter, lambd=0, verbose=False):
    if verbose:
        print("...Generating Kwargs...")
    refractory_period_s = 10 / 30_000
    objective_kwargs = {
        'sampling_rate': int(recording.sampling_frequency),  # sampling rate must be integer
        't_start': 0,
        't_end': int(recording.get_num_samples() / recording.sampling_frequency),  # t_end must be an integer
        'n_sec_chunk': 1,
        'refractory_period_frames': int(refractory_period_s * recording.sampling_frequency),
        'jitter_factor': 8,
        'threshold': 50,
        'lambd' : lambd,
        'verbose': verbose,
    }
    method_kwargs = {
        'objective_kwargs': objective_kwargs,
        'nbefore': nbefore,
        'nafter': nafter,
    }
    return method_kwargs


def shorten_recording(recording, sorting, filepaths, time_range_sec=(0, 10), verbose=False):
    if verbose:
        print("...Shortening Recording...")
    rec_short_path = filepaths['rec_short_path']
    time_range_sample = np.array(time_range_sec) * recording.sampling_frequency
    rec_short = recording.frame_slice(start_frame=time_range_sample[0], end_frame=time_range_sample[1])
    if not rec_short_path.exists():
        rec_short.save(folder=rec_short_path)
    sort_short = sorting.frame_slice(start_frame=time_range_sample[0], end_frame=time_range_sample[1])
    return rec_short, sort_short

def downsample_recording(recording, sorting, templates, we, filepaths, down_factor=1, overwrite=False, verbose=False):
    if verbose:
        print("...Downsampling Recording...")
    rec_down_path = filepaths['rec_down_path']
    sort_down_path = filepaths['sort_down_path']

    # Get traces and downsample
    traces = recording.get_traces()
    traces_downsampled = signal.decimate(traces, down_factor, axis=0)
    fs = recording.sampling_frequency / down_factor # correct sampling frequency for downsampling
    spike_time_inds, spike_ids = sorting.get_all_spike_trains()[0]
    spike_time_inds_down = spike_time_inds // down_factor # correct indices for downsampling
    templates_downsampled = signal.decimate(templates, down_factor, axis=1)
    nbefore = int(np.ceil(we.nbefore / down_factor))
    nafter = int(np.ceil(we.nafter / down_factor))

    # Double-check downsampling
    spike_times = spike_time_inds / recording.sampling_frequency
    spike_times_down = spike_time_inds_down / fs
    epsilon = 1 / fs
    assert np.all(np.abs(spike_times - spike_times_down)<epsilon), "Spike Time Error"
    assert nbefore + nafter == templates_downsampled.shape[1], "nbefore/nafter Error"
    # Generate new recording/sorting and save/load
    rec_down = sc.NumpyRecording(traces_downsampled, fs, channel_ids=recording.channel_ids)
    rec_down.set_probe(recording.get_probe(), in_place=True)
    rec_down.annotate(is_filtered=True)
    if overwrite and rec_down_path.exists():
        shutil.rmtree(rec_down_path)
    if not rec_down_path.exists():
        rec_down.save(folder=rec_down_path)
    rec_down = sc.load_extractor(rec_down_path)
    sorting_down = sc.NumpySorting.from_times_labels(spike_time_inds_down, spike_ids, fs)
    if overwrite and sort_down_path.exists():
        shutil.rmtree(sort_down_path)
    if not sort_down_path.exists():
        sorting_down.save(folder=sort_down_path)
    sorting_down = sc.NpzSortingExtractor.load_from_folder(folder=sort_down_path)

    return rec_down, sorting_down, templates_downsampled, nbefore, nafter

def run_matching(recording, method_kwargs, job_kwargs, verbose=False):
    if verbose:
        print("...Running Matching...")
    job_kwargs['chunk_duration'] = float(method_kwargs['objective_kwargs']['n_sec_chunk'])  # these must be the same
    spikes = find_spikes_from_templates(recording, method='spike-psvae', method_kwargs=method_kwargs, **job_kwargs)
    return spikes


def run_spike_psvae(templates, method_kwargs, filepaths, verbose=False):
    if verbose:
        print("...Running Spike-PSVAE...")
    objective_kwargs = method_kwargs['objective_kwargs']
    objective_kwargs['upsample'] = objective_kwargs.pop('jitter_factor')
    objective_kwargs['template_index_to_unit_id'] = objective_kwargs.pop('template_ids2unit_ids')

    standardized_bin = filepaths['rec_down_path'] / "traces_cached_seg0.raw"
    deconv_path = filepaths['project_path'] / "deconv"
    if not deconv_path.exists():
        deconv_path.mkdir()
    # Instantiate objective and run
    obj = Objective(templates, deconv_path, standardized_bin, **objective_kwargs)
    batch_ids = np.arange(obj.n_batches)
    fnames_out = [deconv_path / f"{i}" for i in batch_ids]
    obj.run(batch_ids, fnames_out)

    # Recover spike trains from saved files
    spiketrain = []
    for batch_id, fname_out in zip(batch_ids, fnames_out):
        fname_out = fname_out.parent / (fname_out.name + '.npz')
        with np.load(fname_out) as d:
            for spike, scaling in zip(d['spike_train'], d['scalings']):
                spike[0] += method_kwargs['nbefore']  # account for 'trough_offset'
                spike[1] //= objective_kwargs['upsample']  # account for multiple upsampled templates
                spiketrain.append(spike)
    spiketrain = np.array(spiketrain)
    spike_dtype = [('sample_ind', 'int64'), ('channel_ind', 'int64'), ('cluster_ind', 'int64'),
                   ('amplitude', 'float64'), ('segment_ind', 'int64')]
    spikes = np.zeros(spiketrain.shape[0], dtype=spike_dtype)
    spikes['sample_ind'] = spiketrain[:, 0]
    spikes['cluster_ind'] = spiketrain[:, 1]
    return spikes


def evaluate_performance(recording, gt_sorting, we, spikes, hit_threshold=0.4e-3, verbose=False):
    if verbose:
        print("...Evaluating Performance...")
    gt_spike_times, gt_spike_ids = gt_sorting.get_all_spike_trains()[0]
    hits, misses, misclasses = 0, 0, 0
    for gt_spike_idx, gt_unit_id in zip(gt_spike_times, gt_spike_ids):
        idx = np.argmin(np.abs(spikes['sample_ind'] - gt_spike_idx))
        spike_idx, unit_id = spikes['sample_ind'][idx], we.unit_ids[spikes['cluster_ind'][idx]]
        time_diff = np.abs(spike_idx - gt_spike_idx) / recording.sampling_frequency
        if time_diff < hit_threshold:
            if gt_unit_id == unit_id:
                hits += 1
            else:
                if verbose:
                    print(f"{time_diff = }")
                    print(f"{gt_unit_id = }")
                    print(f"{unit_id = }")
                misclasses += 1
        else:
            misses += 1
            if verbose:
                print(f"gt_spiketime = {gt_spike_idx / recording.sampling_frequency}")
    hit_rate = hits / len(gt_spike_times)
    misclass_rate = misclasses / len(gt_spike_times)
    miss_rate = misses / len(gt_spike_times)
    return hit_rate, misclass_rate, miss_rate

if __name__ == '__main__':
    test_matching_psvae()
