# Standard Libraries
from pathlib import Path
import numpy as np

# Spikeinterface
import spikeinterface.core as sc
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from spikeinterface.sortingcomponents.matching import find_spikes_from_templates


def test_matching_psvae():
    verbose = False
    filepaths = generate_filepaths(project_name="test_matching_psvae", verbose=verbose)
    job_kwargs = dict(n_jobs=4, chunk_duration='1s', progress_bar=True, verbose=verbose)
    recording, gt_sorting = generate_gt_recording(job_kwargs, filepaths, verbose=verbose)
    gt_templates, we = generate_templates(recording, gt_sorting, job_kwargs, filepaths, verbose=verbose)
    duplicated_templates = np.concatenate((gt_templates, gt_templates[ [0] ]))
    template_ids2unit_ids = list(range(7))
    template_ids2unit_ids.append(0)
    template_ids2unit_ids = np.array(template_ids2unit_ids)
    param_sets = {
        0 : dict(lambd=0, n_jobs=1, template_ids2unit_ids=None, upsample=1),
        1 : dict(lambd=1, n_jobs=2, template_ids2unit_ids=template_ids2unit_ids, upsample=3, vis_su=10),
    }
    for params in param_sets.values():
        print(f"{params = }")
        job_kwargs['n_jobs'] = params.pop('n_jobs')
        if params['template_ids2unit_ids'] is None:
            templates = gt_templates
        else:
            templates = duplicated_templates

        method_kwargs = generate_method_kwargs(recording, templates, we, verbose=verbose)
        method_kwargs['objective_kwargs'].update(params)
        recording, gt_sorting = shorten_recording(recording, gt_sorting, method_kwargs, filepaths, verbose=verbose)
        spikes = run_matching(recording, method_kwargs, job_kwargs, verbose=verbose)
        hit_rate, misclass_rate, miss_rate = evaluate_performance(recording, gt_sorting, we, spikes, verbose=verbose)

        print("Performance:")
        print(f"{hit_rate = }")
        print(f"{misclass_rate = }")
        print(f"{miss_rate = }")
        assert hit_rate == 1, "Missed Spikes"


def generate_filepaths(project_name, verbose=False):
    if verbose:
        print("...Generating Filepaths...")
    base_path = Path("/Volumes/T7/CatalystNeuro")
    mearec_path = base_path / "MEArecDatasets/recordings/toy_recording_30min.h5"
    project_path = base_path / project_name
    rec_path = project_path / "recording"
    rec_short_path = project_path / "rec_short"
    we_path = project_path / "waveforms"
    filepaths = dict(
        base_path = base_path,
        mearec_path = mearec_path,
        project_path = project_path,
        rec_path = rec_path,
        rec_short_path = rec_short_path,
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


def generate_templates(recording, sorting, job_kwargs, filepaths, duplicate_templates=False, verbose=False):
    if verbose:
        print("...Generating Templates...")
    we_kwargs = dict(ms_before=2.5, ms_after=3.5, max_spikes_per_unit=500, load_if_exists=True)
    we = sc.extract_waveforms(recording, sorting, filepaths['we_path'], **we_kwargs, **job_kwargs)
    templates = we.get_all_templates(we.unit_ids)
    return templates, we


def generate_method_kwargs(recording, templates, we, lambd=0, verbose=False):
    if verbose:
        print("...Generating Kwargs...")
    refractory_period_s = 10 / 30_000
    objective_kwargs = {
        'sampling_rate': int(recording.sampling_frequency),  # sampling rate must be integer
        't_start': 0,
        't_end': int(recording.get_num_samples() / recording.sampling_frequency),  # t_end must be an integer
        'n_sec_chunk': 1,
        'refractory_period_frames': int(refractory_period_s * recording.sampling_frequency),
        'upsample': 8,
        'threshold': 50,
        'lambd' : lambd,
        'verbose': verbose,
        'templates': templates,
    }
    method_kwargs = {
        'objective_kwargs': objective_kwargs,
        'nbefore': we.nbefore,
        'nafter': we.nafter,
    }
    return method_kwargs


def shorten_recording(recording, sorting, method_kwargs, filepaths, time_range_sec=(0, 10), verbose=False):
    if verbose:
        print("...Shortening Recording...")
    rec_short_path = filepaths['rec_short_path']
    time_range_sample = np.array(time_range_sec) * recording.sampling_frequency
    rec_short = recording.frame_slice(start_frame=time_range_sample[0], end_frame=time_range_sample[1])
    if not rec_short_path.exists():
        rec_short.save(folder=rec_short_path)
    method_kwargs['objective_kwargs']['t_end'] = time_range_sec[1]
    sort_short = sorting.frame_slice(start_frame=time_range_sample[0], end_frame=time_range_sample[1])
    return rec_short, sort_short


def run_matching(recording, method_kwargs, job_kwargs, verbose=False):
    if verbose:
        print("...Running Matching...")
    job_kwargs['chunk_duration'] = float(method_kwargs['objective_kwargs']['n_sec_chunk'])  # these must be the same
    spikes = find_spikes_from_templates(recording, method='spike-psvae', method_kwargs=method_kwargs, **job_kwargs)
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
                print(f"{time_diff = }")
                print(f"{gt_unit_id = }")
                print(f"{unit_id = }")
                misclasses += 1
        else:
            misses += 1
            print(f"gt_spiketime = {gt_spike_idx / recording.sampling_frequency}")
    hit_rate = hits / len(gt_spike_times)
    misclass_rate = misclasses / len(gt_spike_times)
    miss_rate = misses / len(gt_spike_times)
    return hit_rate, misclass_rate, miss_rate

if __name__ == '__main__':
    test_matching_psvae()
