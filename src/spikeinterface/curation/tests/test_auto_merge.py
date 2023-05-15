import pytest
import shutil
from pathlib import Path
import numpy as np

from spikeinterface import WaveformExtractor, load_extractor, extract_waveforms, NumpySorting, set_global_tmp_folder
from spikeinterface.extractors import toy_example


from spikeinterface.core.generate import inject_some_split_units
from spikeinterface.curation import get_potential_auto_merge
from spikeinterface.curation.auto_merge import normalize_correlogram


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "curation"
else:
    cache_folder = Path("cache_folder") / "curation"

set_global_tmp_folder(cache_folder)

def test_get_auto_merge_list():


    rec, sorting = toy_example(num_segments=1, num_units=5, duration=[300.], firing_rate=20., seed=0)
    
    num_unit_splited = 1
    num_split = 2

    sorting_with_split, other_ids = inject_some_split_units(sorting, split_ids=sorting.unit_ids[:num_unit_splited], num_split=num_split, output_ids=True)
    

    # print(sorting_with_split)
    # print(sorting_with_split.unit_ids)

    rec = rec.save()
    sorting_with_split = sorting_with_split.save()
    wf_folder = cache_folder / 'wf_auto_merge'
    if wf_folder.exists():
        shutil.rmtree(wf_folder)
    we = extract_waveforms(rec, sorting_with_split, mode='folder', folder=wf_folder, n_jobs=1)

    # we = extract_waveforms(rec, sorting_with_split, mode='memory', folder=None, n_jobs=1)
    # print(we)

    potential_merges, outs = get_potential_auto_merge(we,
                minimum_spikes=1000, maximum_distance_um=150.,
                peak_sign="neg",
                bin_ms=0.25, window_ms=100.,
                corr_diff_thresh=0.16,
                template_diff_thresh=0.25,
                censored_period_ms=0., refractory_period_ms=4.0,
                sigma_smooth_ms = 0.6,
                contamination_threshold=0.2,
                adaptative_window_threshold=0.5,
                num_channels=5,
                num_shift=5,
                firing_contamination_balance=1.5,
                extra_outputs=True,
                )
    # print(potential_merges)

    assert len(potential_merges) == num_unit_splited
    for true_pair in other_ids.values():
        true_pair = tuple(true_pair)
        assert true_pair in potential_merges
        
        
    


    # import matplotlib.pyplot as plt
    # templates_diff = outs['templates_diff']
    # correlogram_diff = outs['correlogram_diff']
    # bins = outs['bins']
    # correlograms_smoothed = outs['correlograms_smoothed']
    # correlograms = outs['correlograms']
    # win_sizes = outs['win_sizes']

    # fig, ax = plt.subplots()
    # ax.hist(correlogram_diff.flatten(), bins=np.arange(0, 1, 0.05))

    # fig, ax = plt.subplots()
    # ax.hist(templates_diff.flatten(), bins=np.arange(0, 1, 0.05))

    # m = correlograms.shape[2] // 2

    # for unit_id1, unit_id2 in potential_merges[:5]:
        # unit_ind1 = sorting_with_split.id_to_index(unit_id1)
        # unit_ind2 = sorting_with_split.id_to_index(unit_id2)

        # bins2 = bins[:-1] + np.mean(np.diff(bins))
        # fig, axs = plt.subplots(ncols=3)
        # ax = axs[0]
        # ax.plot(bins2, correlograms[unit_ind1, unit_ind1, :], color='b')
        # ax.plot(bins2, correlograms[unit_ind2, unit_ind2, :], color='r')
        # ax.plot(bins2, correlograms_smoothed[unit_ind1, unit_ind1, :], color='b')
        # ax.plot(bins2, correlograms_smoothed[unit_ind2, unit_ind2, :], color='r')
        
        # ax.set_title(f'{unit_id1} {unit_id2}')
        # ax = axs[1]
        # ax.plot(bins2, correlograms_smoothed[unit_ind1, unit_ind2, :], color='g')

        # auto_corr1 = normalize_correlogram(correlograms_smoothed[unit_ind1, unit_ind1, :])
        # auto_corr2 = normalize_correlogram(correlograms_smoothed[unit_ind2, unit_ind2, :])
        # cross_corr = normalize_correlogram(correlograms_smoothed[unit_ind1, unit_ind2, :])

        # ax = axs[2]
        # ax.plot(bins2, auto_corr1, color='b')
        # ax.plot(bins2, auto_corr2, color='r')
        # ax.plot(bins2, cross_corr, color='g')

        # ax.axvline(bins2[m - win_sizes[unit_ind1]], color='b')
        # ax.axvline(bins2[m + win_sizes[unit_ind1]], color='b')
        # ax.axvline(bins2[m - win_sizes[unit_ind2]], color='r')
        # ax.axvline(bins2[m + win_sizes[unit_ind2]], color='r')
        
        # ax.set_title(f'corr diff {correlogram_diff[unit_ind1, unit_ind2]} - temp diff {templates_diff[unit_ind1, unit_ind2]}')
    # plt.show()


    
if __name__ == '__main__':
    test_get_auto_merge_list()
