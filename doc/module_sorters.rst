Sorters module
==============

The :py:mod:`spikeinterface.sorters` module is where spike sorting happens!

SpikeInterface provides wrapper classes to many commonly used spike sorters (see :ref:`compatible-tech`).
All sorter classes inherit from the :py:class:`~spikeinterface.sorters.BaseSorter` class, which provides the
common tools to run spike sorters.


Sorter wrappers concept
-----------------------

Each spike sorter wrapper includes:

* a list of default parameters
* a list of parameter description
* a :code:`_setup_recording` class function, which parses the required files and metadata for each sorter into the specified :code:`output_folder`
* a :code:`_run_from_folder` class function, which launches the spike sorter from the :code:`output_folder`
* a :code:`_get_result_from_folder` class function, which loads the :code:`SortingExtractor` from the :code:`output_folder`


Example
-------

The :code:`sorters` includes :code:`run()` functions to easily run spike sorters:

.. code-block:: python

    import spikeinterface.sorters as ss

    # recording is a RecordingExtractor object
    sorting_TDC = ss.run_tridesclous(recording, output_folder="tridesclous_output")

    # which is equivalent to
    sorting_TDC = ss.run_sorter("tridesclous", recording, output_folder="tridesclous_output")


Running sorters in container docker/singularity
-----------------------------------------------

One of the biggest bottlenecks for users is installing spike sorting software. To alleviate this, we build and
maintain containerized versions of several popular spike sorters on the `SpikeInterface Docker Hub repository
<https://hub.docker.com/u/spikeinterface>`_. 

See full documentation here: :ref:`containerizedsorters`

Run several sorting jobs in parallel
------------------------------------

The :py:mod:`spikeinterface.sorters` includes also includes tools to run several spike sorting jobs in parallel. This
can be done with the :py:func:`spikeinterface.sorters.run_sorters()` function by specifying an :code:`engine` that
supports parallel processing (e.g. joblib or dask).

In this code example, 3 sorters are run on 2 recordings using 6 jobs:

.. code-block:: python

    import spikeinterface.sorters as ss

    # recording1 and recording2 are RecordingExtractor objects
    recording_dict = {"rec1": recording1, "rec2": recording2}

    sorting_outputs = ss.run_sorters(
        sorter_list=["tridesclous", "herdingspikes", "ironclust"],
        recording_dict_or_list=recording_dict,
        working_folder="all_sorters",
        verbose=False,
        engine="joblib",
        engine_kwargs={'n_jobs': 6},
    )

After the jobs are run, the :code:`sorting_outputs` is a dictionary with :code:`(rec_name, sorter_name)` as key (e.g.
:code:`('rec1', 'tridesclous')` in this example), and the corresponding :code:`SortingExtractor` as value.


