Curation module
===============

**Note** In January 2023, this module is still under construction and quite experimental. The api
of some function could be changed/improved from time to time.

Manual curation
---------------

spikeinterface offer a machinery to manually clean a sorting and keep track of the cleaning history.
The cleaning have several "steps" that can be be repeated and chained:
  * remove/select units
  * split units
  * merge units

This functionality is done with :py:class:`~spikeinterface.curation.CurationSorting`.
Internaly this class keep the history as a graph. Also internally split and merges are handle by
:py:class:`~spikeinterface.curation.MergeUnitsSorting` and 
:py:class:`~spikeinterface.curation.SplitUnitSorting`. Theses 2 classes can also be used independently.



.. code-block:: python

    from spikeinterface.curation import CurationSorting

    sorting = run_sorter('kilosort', recording)

    cs = CurationSorting(sorting)

    # make a first merge
    cs.merge(['#1', '#5', '#15'])

    # make a second merge
    cs.merge(['#11', '#21'])

    # make a split
    split_index = ... # some creteria on spikes
    cs.split('#20', split_index)

    # here the final clean sorting
    clean_sorting = cs.sorting





Automatic curation tools
------------------------

`Lussac <https://www.biorxiv.org/content/10.1101/2022.02.08.479192v1>`_ is an external package for auto clean the
results of a sorting.

Some concept like auto-merging have ported to spikeinterface.

:py:func:`~spikeinterface.curation.get_potential_auto_merge` give a list of potential merges.
Then this list can be applied to the sorting.

:py:func:`~spikeinterface.curation.get_potential_auto_merge` have many internal trick and steps and so have many option.
This must be carffuly choosen and of course not applied blindly!



.. code-block:: python

    from spikeinterface.curation import MergeUnitsSorting, get_potential_auto_merge

    sorting = run_sorter('kilosort', recording)

    we = extract_waveforms(recording, sorting, folder='/wf_folder')

    # merges is a list of list of unit_ids to be merged.
    merges = get_potential_auto_merge(we, minimum_spikes=1000,  maximum_distance_um=150.,
                                      peak_sign="neg", bin_ms=0.25, window_ms=100.,
                                      corr_diff_thresh=0.16, template_diff_thresh=0.25,
                                      censored_period_ms=0., refractory_period_ms=1.0,
                                      contamination_threshold=0.2, num_channels=5, num_shift=5,
                                      firing_contamination_balance=1.5,
                                      extra_outputs=False)

    # here we apply the merges
    clean_sorting = MergeUnitsSorting(sorting, merges)


Manual curation with sorting view
---------------------------------

:code:`sortingview` expose a powerfull GUI inside the browser in spikeinterface we have a simple machinery
to export the sorting to this we-based manual curation tools and also the machinery to retrieve the curation
and apply it a to a sorting to clean it.



.. code-block:: python


    from spikeinterface.curation import apply_sortingview_curation
    from spikeinterface.widgets import plot_sorting_summary

    # run a sorter and export waveforms
    sorting = run_sorter('kilosort', recording)
    we = extract_waveforms(recording, sorting, folder='/wf_folder')    

    # this push the cloud data for plot
    url = plot_sorting_summary(we, backend='sortingview')
    # we open the url in firefox
    # make manual merges/split/remove
    # On the curation box click on "Save as snapshot (sha1://)"

    # we copy back the uri
    sha_uri = "sha1://59feb326204cf61356f1a2eb31f04d8e0177c4f1"
    clean_sorting = apply_sortingview_curation(sorting, uri_or_json=sha_uri)

    # Note : this could be done done with a json file



Other curation tools
--------------------

We have other tools for manual action of curation

 * :py:func:`~spikeinterface.curation.find_duplicated_spikes` : find duplicated spike train on one spiketrain
 * :py:func:`~spikeinterface.curation.remove_duplicated_spikes` : remove all duplicated spike from a sorting
   using inetrnaly the previous function

