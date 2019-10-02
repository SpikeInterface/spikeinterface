"""
This build static figures  with dataset from
the google drive : "study_mearec_SqMEA1015um".




"""

import spikeextractors as se
import spikecomparison as sc
import spikewidgets as sw

import numpy as np
import matplotlib.pyplot as plt


def make_comparison_figures():
    
    gt_sorting = se.NpzSortingExtractor('rec0.npz')
    tested_sorting = se.NpzSortingExtractor('rec0[#]spykingcircus.npz')
    print('gt_sorting', len(gt_sorting.get_unit_ids()))
    print('tested_sorting', len(tested_sorting.get_unit_ids()))
    
    comp = sc.compare_sorter_to_ground_truth(gt_sorting, tested_sorting, gt_name=None, tested_name=None,
                                   delta_time=0.4, sampling_frequency=None, min_accuracy=0.5, exhaustive_gt=True, match_mode='hungarian', 
                                   n_jobs=-1, bad_redundant_threshold=0.2, compute_labels=False, verbose=False)
    
    print(comp.hungarian_match_12)
    
    exit()
    
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.matshow(comp.match_event_count, cmap='Greens')
    ax.set_xticks(np.arange(0, comp.match_event_count.shape[1]))
    ax.set_yticks(np.arange(0, comp.match_event_count.shape[0]))
    ax.xaxis.tick_bottom()
    ax.set_yticklabels(comp.match_event_count.index, fontsize=12)
    ax.set_xticklabels(comp.match_event_count.columns, fontsize=12)
    fig.colorbar(im)
    fig.savefig('spikecomparison_match_count.png')
    
    fig, ax = plt.subplots(figsize=(15, 10))
    sw.plot_agreement_matrix(comp, ax=ax, ordered=False)
    im = ax.get_images()[0]
    fig.colorbar(im)
    fig.savefig('spikecomparison_agreement_unordered.png')

    fig, ax = plt.subplots(figsize=(15, 10))
    sw.plot_agreement_matrix(comp, ax=ax)
    im = ax.get_images()[0]
    fig.colorbar(im)
    fig.savefig('spikecomparison_agreement.png')
    
    fig, ax = plt.subplots(figsize=(15, 10))
    sw.plot_confusion_matrix(comp, ax=ax)
    im = ax.get_images()[0]
    fig.colorbar(im)
    fig.savefig('spikecomparison_confusion.png')
    
    
    
    
    
    plt.show()
    
    
    
    
    
if __name__ == '__main__':
    make_comparison_figures()