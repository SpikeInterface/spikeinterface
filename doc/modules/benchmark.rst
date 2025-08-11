Benchmark module
================

This module contains machinery to compare some sorters against ground truth in multiple situations.

..notes::

    In 0.102.0 The previous :py:func:`~spikeinterface.comparison.GroundTruthStudy()` has been replaced by
    :py:func:`~spikeinterface.benchmark.SorterStudy()`


This module also aims to benchmark sorting components (detection, clustering, motion, template matching) using the
same base class :py:func:`~spikeinterface.benchmark.BenchmarkStudy()` but specialized to a targeted component.

By design, the main class handles the concept of "levels" : this allows you to compare several complexities at the same time.
For instance, compare kilosort4 vs kilsort2.5 (level 0) for different noises amplitudes (level 1) combined with
several motion vectors (level 2).

**Example: compare many sorters : a ground truth study**

We have a high level class to compare many sorters against ground truth: :py:func:`~spikeinterface.benchmark.SorterStudy()`

A study is a systematic performance comparison of several ground truth recordings with several sorters or several cases,
like the different parameter sets.

The study class proposes high-level tool functions to run many ground truth comparisons with many "cases"
on many recordings and then collect and aggregate results in an easy way.

The mechanism is based on an intrinsic organization into a "study_folder" with several subfolders:

  * datasets: contains ground truth datasets
  * sorters : contains outputs of sorters
  * sortings: contains light copy of all sorting
  * metrics: contains metrics
  * ...


.. code-block:: python

    import spikeinterface as si
    import spikeinterface.widgets as sw
    from spikeinterface.benchmark import SorterStudy

    # generate 2 simulated datasets (could be also mearec files)
    rec0, gt_sorting0 = si.generate_ground_truth_recording(num_channels=4, durations=[30.], seed=42)
    rec1, gt_sorting1 = si.generate_ground_truth_recording(num_channels=4, durations=[30.], seed=91)

    datasets = {
        "toy0": (rec0, gt_sorting0),
        "toy1": (rec1, gt_sorting1),
    }

    # define some "cases". Here we want to test tridesclous2 on 2 datasets and spykingcircus2 on one dataset
    # so it is a two level study (sorter_name, dataset)
    # this could be more complicated like (sorter_name, dataset, params)
    cases = {
        ("tdc2", "toy0"): {
            "label": "tridesclous2 on tetrode0",
            "dataset": "toy0",
            "params": {"sorter_name": "tridesclous2"}
        },
        ("tdc2", "toy1"): {
            "label": "tridesclous2 on tetrode1",
            "dataset": "toy1",
            "params": {"sorter_name": "tridesclous2"}
        },
        ("sc", "toy0"): {
            "label": "spykingcircus2 on tetrode0",
            "dataset": "toy0",
            "params": {
                "sorter_name": "spykingcircus2",
                "docker_image": True
            },
        },
    }
    # this initializes a folder
    study_folder = "my_study_folder"
    study = SorterStudy.create(study_folder=study_folder, datasets=datasets, cases=cases,
                                    levels=["sorter_name", "dataset"])


    # This internally does run_sorter() for all cases in one function
    study.run()

    # Run the benchmark : this internally does compare_sorter_to_ground_truth() for all cases
    study.compute_results()

    # Collect comparisons one by one
    for case_key in study.cases:
        print('*' * 10)
        print(case_key)
        # raw counting of tp/fp/...
        comp = study.get_result(case_key)["gt_comparison"]
        # summary
        comp.print_summary()
        perf_unit = comp.get_performance(method='by_unit')
        perf_avg = comp.get_performance(method='pooled_with_average')
        # some plots
        m = comp.get_confusion_matrix()
        w_comp = sw.plot_agreement_matrix(sorting_comparison=comp)

    # Collect synthetic dataframes and display.
    # As shown previously, the performance is returned as a pandas dataframe.
    # The spikeinterface.comparison.get_performance_by_unit() function
    # gathers all the outputs in the study folder and merges them into a single dataframe.
    # Same idea for spikeinterface.comparison.get_count_units()

    # this is a dataframe
    perfs = study.get_performance_by_unit()

    # this is a dataframe
    unit_counts = study.get_count_units()

    # Study also has several plotting methods for plotting the result
    study.plot_agreement_matrix()
    study.plot_unit_counts()
    study.plot_performances(mode="ordered")


Benchmark spike collisions
--------------------------

SpikeInterface also has a specific toolset to benchmark how well sorters are at recovering spikes in "collision".

We have three classes to handle collision-specific comparisons, and also to quantify the effects on correlogram
estimation:

  * :py:class:`~spikeinterface.comparison.CollisionGTComparison`
  * :py:class:`~spikeinterface.comparison.CorrelogramGTComparison`

For more details, checkout the following paper:

`Samuel Garcia, Alessio P. Buccino and Pierre Yger. "How Do Spike Collisions Affect Spike Sorting Performance?" <https://doi.org/10.1523/ENEURO.0105-22.2022>`_
