Sorters module
==============

The :py:mod:`spikeinterface.sorters` module is where spike sorting happens!

TODO : exaplain external vs internal

SpikeInterface provides wrapper classes to many commonly used spike sorters (see :ref:`compatible-sorters`).
All sorter classes inherit from the :py:class:`~spikeinterface.sorters.BaseSorter` class, which provides the
common tools to run spike sorters.

TODO add a world about the installation and docker/singularity


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


.. _compatible-sorters:

Supported Spike Sorters
-----------------------

Currently, we support many popular semi-automatic spike sorters.  Given the standardized, modular design of our sorters, adding new ones is straightforward so we expect this list to grow in future versions.


* **HerdingSpikes2** - HerdingspikesSorter
* **IronClust** - IronClustSorter
* **Kilosort** - KilosortSorter
* **Kilosort2** - Kilosort2Sorter
* **Kilosort2.5** - Kilosort2_5Sorter
* **Kilosort3** - Kilosort3Sorter
* **PyKilosort** - PyKilosortSorter
* **Klusta** - KlustaSorter
* **Mountainsort4** - Mountainsort4Sorter
* **SpyKING Circus** - SpykingcircusSorter
* **Tridesclous** - TridesclousSorter
* **Wave clus** - WaveClusSorter
* **Combinato** - CombinatoSorter
* **HDSort** - HDSortSorter
* **yass** - YassSorter


Installed Sorters
-----------------

To check which sorters are useable in a given python environment, one can print the installed sorters list. An example is shown in a pre-defined miniconda3 environment.

First, import the spikesorters package,

.. code:: python

  import spikeinterface.sorters as ss

Then you can check the installed Sorter list,

.. code:: python

  ss.installed_sorters()

which outputs,

.. parsed-literal::
  ['herdingspikes',
   'klusta',
   'mountainsort4',
   'spykingcircus',
   'tridesclous']


When trying to use an sorter that has not been installed in your environment, an installation message will appear indicating how to install the given sorter,

.. code:: python

  recording = ss.run_ironclust(recording)

throws the error,

.. parsed-literal::
  AssertionError: This sorter ironclust is not installed.
        Please install it with:

  To use IronClust run:

        >>> git clone https://github.com/jamesjun/ironclust
    and provide the installation path by setting the IRONCLUST_PATH
    environment variables or using IronClustSorter.set_ironclust_path().




.. _containerizedsorters:

Running sorters in container docker/singularity
-----------------------------------------------

One of the biggest bottlenecks for users is installing spike sorting software. To alleviate this, we build and
maintain containerized versions of several popular spike sorters on the `SpikeInterface Docker Hub repository
<https://hub.docker.com/u/spikeinterface>`_. 

See full documentation here: :ref:`containerizedsorters`



One of the biggest bottlenecks for users is installing spike sorting software. To alleviate this, we build and
maintain containerized versions for most spike sorters.

The containerized approach has several advantages:  

* Installation is much easier.  
* Different spike sorters with conflicting dependencies can be easily run side-by-side.  
* The results of the analysis are more reproducible and not dependant on the operating system  
* MATLAB-based sorters can be run without a MATLAB licence.  

The containers can be run in Docker or Singularity, so having Docker or Singularity installed is a prerequisite.


Running spike sorting in a docker container container just requires to:

1) have docker installed
2) have docker python SDK installed (:code:`pip install docker`)

or

1) have singularity installed
2) have `singularity python <https://singularityhub.github.io/singularity-cli/>`_ (:code:`pip install spython`)

Some sorters are GPU required or optional. To run containerized sorters with GPU capabilities, CUDA and `nvidia-container-toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_ needs to be installed.
Only NVIDIA GPUs are supported for now.

For Docker users, you can either install `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_ 
(recommended for Windows and MacOS) or the `Docker Engine  <https://docs.docker.com/engine/install/ubuntu/>`_ 
(recommended for Linux). 
To enable :code:`Docker Desktop` to download the containers, you need to create an account on 
`DockerHub <https://hub.docker.com/>`_ (free) and perform the login in :code:`Docker Desktop`.
For :code:`Docker Engine`, you also need to enable Docker to run without :code:`sudo` priviledges 
following `this post-install guide <https://docs.docker.com/engine/install/linux-postinstall/>`_

The containers are built with Docker, but Singularity has an internal mechanism to convert docker images.
Using Singularity is often prefered due to its simpler approach with regard to root privilege.

The following code creates a test recording and runs a containerized spike sorter (Kilosort 3):

.. code-block:: python

    import spikeinterface.extractors as se
    import spikeinterface.sorters as ss
    test_recording, _ = se.toy_example(
        duration=30,
        seed=0,
        num_channels=64,
        num_segments=1
    )
    test_recording = test_recording.save(folder="test-docker-folder")

    sorting = ss.run_kilosort3(
        recording=test_recording,
        output_folder="kilosort3",
        singularity_image=True)

    print(sorting)


This will automatically check if the latest compiled kilosort3 docker image is present on your workstation and if it is not the proper image will be downloaded from `SpikeInterface's Docker Hub repository <https://hub.docker.com/u/spikeinterface>`_. The sorter will then run and output the results in the designated folder. 

To run in Docker instead of Singularity, use ``docker_image=True``. 

.. code-block:: python

    sorting = ss.run_kilosort3(recording=test_recording, output_folder="kilosort3", docker_image=True)

To use a specific image, set either ``docker_image`` or ``singularity_image`` to a string, e.g. ``singularity_image="spikeinterface/kilosort3-compiled-base:0.1.0"``.

.. code-block:: python

    sorting = ss.run_kilosort3(
        recording=test_recording,
        output_folder="kilosort3",
        singularity_image="spikeinterface/kilosort3-compiled-base:0.1.0")


**NOTE:** the :code:`toy_example()` returns in-memory objects, which are not bound to a file on disk. 
In order to run spike sorting in a container, the recording object MUST be persistent on disk, so that the 
container can reload it. The :code:`save()` function makes the recording persistent on disk, by saving the in-memory 
:code:`test_recording` object to a binary file in the :code:`test-docker-folder` folder.


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


run sorter on separate groups
-----------------------------

TODO concept

TODO example

TODO agglomerate back several sortings



Internal sorters
----------------

TODO concept

TODO example

TODO dependencies



Contributing
------------

The containerization of spike sorters is managed on a separate GitHub repo, `spikeinterface-dockerfiles
<https://github.com/SpikeInterface/spikeinterface-dockerfiles>`_. 
If you find an error with a current container or would like to request a new spike sorter, please submit an Issue to this repo.
