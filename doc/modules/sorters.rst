Sorters module
==============

The :py:mod:`spikeinterface.sorters` module is where spike sorting happens!

On one hand, SpikeInterface provides wrapper classes to many commonly used spike sorters like
Kilosort, Spyking-circus, etc. (see :ref:`compatible-sorters`). All these sorter classes inherit
from the :py:class:`~spikeinterface.sorters.BaseSorter` class, which provides the common tools to
run spike sorters.

On the other hand SpikeInterface directly implements some internal sorters (**spykingcircus2**) 
that do not depend on external tools, but depend on the :py:mod:`spikeinterface.sortingcomponents` 
module. **Note that internal sorters are currently experimental and under development**.

A drawback of using external sorters is the installation of these tools. Sometimes they need MATLAB, 
specific versions of CUDA, specific gcc versions vary or even worse outdated versions of
Python/NumPy. In that case, SpikeInterface offer the mechanism of running external sorters inside a
container (docker/singularity) with the sorter pre-installed. See :ref:`containerizedsorters`.


External sorters: the "wrapper" concept
---------------------------------------

When running external sorters, we use the concept of "wrappers". In short, we have some Python code
that generates the external code needed to run the sorter (for instance, a MATLAB script) and also 
external configuration files. Then, the generated code is run in the background with the appropriate 
tools (e.g., Python, MATLAB, Command Line Interfaces).
When the spike sorting process is finished, the output is loaded back in Python into a 
the :py:class:`~spikeinterface.core.BaseSorting` object.

For instance, we the :code:`Kilosort2_5Sorter` will handle:
  * Formatting the data and parameters for Kilosort2.5, using :code:`Kilosort2_5Sorter.setup_recording()`
  * Running MATLAB and Kilosort2.5 code in the folder, using :code:`Kilosort2_5Sorter.run_from_folder()`
  * Retriving the spike sorting output, using :code:`Kilosort2_5Sorter.get_result_from_folder()`

From the user's point of view all of this is in the background and it happens automatically when using the 
:py:func:`~spikeinterface.sorters.run_sorter` function.



Running different spike sorters
-------------------------------

The :py:func:`~spikeinterface.sorters` includes :py:func:`~spikeinterface.sorters.run_sorter` function
to easily run spike sorters:

.. code-block:: python

    from spikeinterface.sorters import run_sorter

    # run tridesclous
    sorting_TDC = run_sorter("tridesclous", recording, output_folder="/folder_TDC")
    # run kilosort.5
    sorting_KS2_5 = run_sorter("kilosort2_5", recording, output_folder="/folder_KS2.5")
    # run IronClust
    sorting_IC = run_sorter("ironclust", recording, output_folder="/folder_IC")
    # run pykilosort
    sorting_pyKS = run_sorter("pykilosort", recording, output_folder="/folder_pyKS")
    # run spykingcircus
    sorting_SC = run_sorter("ironclust", recording, output_folder="/folder_SC")


Then the output, which is a :py:class:`~spikeinterface.core.BaseSorting` object, can be easily 
saved or directly post-processed:

.. code-block:: python

    sorting_TDC.save(folder='/path/to/tridescloud_sorting_output')


The :py:func:`~spikeinterface.sorters.run_sorter` function has several options:

  * to remove or not the sorter working folder (:code:`output_folder/sorter_output`) 
    with: :code:`remove_existing_folder=True/False` (this can save lot of space because some sorters
    need data duplication!)
  * to control ther verbosity: :code:`verbose=False/True`
  * to raise/not raise errors (if they fail): :code:`raise_error=False/True`

Spike-sorter-specific parameters can be controlled directly from the 
:py:func:`~spikeinterface.sorters.run_sorter` function:

.. code-block:: python

    sorting_TDC = run_sorter('tridesclous', recording, output_folder="/folder_TDC",
                             detect_threshold=8.)

    sorting_KS2_5 = run_sorter("kilosort2_5", recording, output_folder="/folder_KS2.5"
                               do_correction=False, preclust_threshold=6, freq_min=200.)


Parameters from all sorters can be retrieved with these functions:

.. code-block:: python

    params = get_default_sorter_params('spykingcircus')
    print("Parameters:\n", params)

    desc = get_sorter_params_description('spykingcircus')
    print("Descriptions:\n", desc)

.. parsed-literal::

    Parameters:
    {'adjacency_radius': 100,
    'auto_merge': 0.75,
    'clustering_max_elts': 10000,
    'detect_sign': -1,
    'detect_threshold': 6,
    'filter': True,
    'merge_spikes': True,
    'num_workers': None,
    'template_width_ms': 3,
    'whitening_max_elts': 1000}

    Descriptions:
    {'adjacency_radius': 'Radius in um to build channel neighborhood',
    'auto_merge': 'Automatic merging threshold',
    'clustering_max_elts': 'Max number of events per electrode for clustering',
    'detect_sign': 'Use -1 (negative), 1 (positive) or 0 (both) depending on the '
                    'sign of the spikes in the recording',
    'detect_threshold': 'Threshold for spike detection',
    'filter': 'Enable or disable filter',
    'merge_spikes': 'Enable or disable automatic mergind',
    'num_workers': 'Number of workers (if None, half of the cpu number is used)',
    'template_width_ms': 'Template width in ms. Recommended values: 3 for in vivo '
                          '- 5 for in vitro',
    'whitening_max_elts': 'Max number of events per electrode for whitening'}


.. _containerizedsorters:

Running sorters in container docker/singularity
-----------------------------------------------

One of the biggest bottlenecks for users is installing spike sorting software. To alleviate this,
we build and maintain containerized versions of several popular spike sorters on the
`SpikeInterface Docker Hub repository <https://hub.docker.com/u/spikeinterface>`_. 

The containerized approach has several advantages:  

* Installation is much easier.  
* Different spike sorters with conflicting dependencies can be easily run side-by-side.  
* The results of the analysis are more reproducible and not dependant on the operating system  
* MATLAB-based sorters can be run **without a MATLAB licence**.

The containers can be run in Docker or Singularity, so having Docker or Singularity installed
is a prerequisite.


Running spike sorting in a docker container just requires:

1) have docker installed
2) have docker Python SDK installed (:code:`pip install docker`)

or

1) have singularity installed
2) have `singularity python <https://singularityhub.github.io/singularity-cli/>`_ (:code:`pip install spython`)

Some sorters are GPU required or optional. To run containerized sorters with GPU capabilities,
CUDA and `nvidia-container-toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_
needs to be installed. Only NVIDIA GPUs are supported for now.


For Docker users, you can either install `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_ 
(recommended for Windows and MacOS) or the `Docker Engine  <https://docs.docker.com/engine/install/ubuntu/>`_ 
(recommended for Linux). 
To enable :code:`Docker Desktop` to download the containers, you need to create an account on 
`DockerHub <https://hub.docker.com/>`_ (free) and perform the login in :code:`Docker Desktop`.
For :code:`Docker Engine`, you also need to enable Docker to run without :code:`sudo` privileges 
following `this post-install guide <https://docs.docker.com/engine/install/linux-postinstall/>`_

The containers are built with Docker, but Singularity has an internal mechanism to convert docker images.
Using Singularity is often preferred due to its simpler approach with regard to root privilege.

The following code creates a test recording and runs a containerized spike sorter (Kilosort 3):

.. code-block:: python

    test_recording, _ = toy_example(
        duration=30,
        seed=0,
        num_channels=64,
        num_segments=1
    )
    test_recording = test_recording.save(folder="test-docker-folder")

    sorting = ss.run_sorter('kilosort3',
        recording=test_recording,
        output_folder="kilosort3",
        singularity_image=True)

    print(sorting)

This will automatically check if the latest compiled kilosort3 docker image is present on your
workstation and if it is not the proper image will be downloaded from
`SpikeInterface's Docker Hub repository <https://hub.docker.com/u/spikeinterface>`_.
The sorter will then run and output the results in the designated folder. 

To run in Docker instead of Singularity, use ``docker_image=True``. 

.. code-block:: python

    sorting = run_sorter('kilosort3', recording=test_recording,
                         output_folder="/tmp/kilosort3", docker_image=True)

To use a specific image, set either ``docker_image`` or ``singularity_image`` to a string, 
e.g. ``singularity_image="spikeinterface/kilosort3-compiled-base:0.1.0"``.

.. code-block:: python

    sorting = run_sorter("kilosort3",
        recording=test_recording,
        output_folder="kilosort3",
        singularity_image="spikeinterface/kilosort3-compiled-base:0.1.0")


**NOTE:** the :code:`toy_example()` returns in-memory objects, which are not bound to a file on disk. 
In order to run spike sorting in a container, the recording object MUST be persistent on disk, so
that the container can reload it. The :code:`save()` function makes the recording persistent on disk,
by saving the in-memory  :code:`test_recording` object to a binary file in the
:code:`test-docker-folder` folder.


Running several sorters in parallel
-----------------------------------

The :py:mod:`~spikeinterface.sorters` module also includes tools to run several spike sorting jobs 
sequentially or in parallel. This can be done with the 
:py:func:`~spikeinterface.sorters.run_sorters()` function by specifying
an :code:`engine` that supports parallel processing (such as :code:`joblib` or :code:`slurm`).

.. code-block:: python

    recordings = {'rec1' : recording, 'rec2': another_recording}
    sorter_list = ['herdingspikes', 'tridesclous']
    sorter_params = { 
                    'herdingspikes': {'clustering_bandwidth' : 8},
                    'tridesclous': {'detect_threshold' : 5.},
                    }
    sorting_output = run_sorters(sorter_list, recordings, working_folder='tmp_some_sorters', 
                                 mode_if_folder_exists='overwrite', sorter_params=sorter_params)

    # the output is a dict with (rec_name, sorter_name) as keys
    for (rec_name, sorter_name), sorting in sorting_output.items():
        print(rec_name, sorter_name, ':', sorting.get_unit_ids())

After the jobs are run, the :code:`sorting_outputs` is a dictionary with :code:`(rec_name, sorter_name)` as key (e.g.
:code:`('rec1', 'tridesclous')` in this example), and the corresponding :py:class:`~spikeinterface.core.BaseSorting` 
as value.

:py:func:`~spikeinterface.sorters.run_sorters` have several "engines" available to launch the computation:

* "loop": sequential 
* "joblib": in parallel
* "slurm": in parallel, using the SLURM job manager

.. code-block:: python

  run_sorters(sorter_list, recordings, engine='loop')

  run_sorters(sorter_list, recordings, engine='joblib', 
              engine_kwargs={'n_jobs': 2})

  run_sorters(sorter_list, recordings, engine='slurm',
              engine_kwargs={'cpus_per_task': 10, 'mem', '5G'})


Spike sorting by group
----------------------

Sometimes you may want to spike sort using specific grouping, for example when working with tetrodes, with multi-shank
probes, or if the recording has data from different probes.
Alternatively, for long silicon probes, such as Neuropixels, one could think of spike sorting different areas 
separately, for example using a different sorter for the hippocampus, the thalamus, or the cerebellum. 
Running spike sorting by group is indeed a very common need.

A :py:class:`~spikeinterface.core.BaseRecording` object has the ability to split itself into a dictionary of 
sub-recordings given a certain property (see :py:meth:`~spikeinterface.core.BaseRecording.split_by`).
So it is easy to loop over this dictionary and sequentially run spike sorting on these sub-recordings.
SpikeInterface also proposes a high-level function to automate the process of splitting the
recording and then aggregating the results with the :py:func:`~spikeinterface.sorters.run_sorter_by_property` function.

In this example, we create a 16-channel recording with 4 tetrodes:

.. code-block:: python

    recording, _ = se.toy_example(duration=[10.], num_segments=1, num_channels=16)
    print(recording.get_channel_groups())
    # >>> [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

    # create 4 tetrodes
    from probeinterface import generate_tetrode, ProbeGroup
    probegroup = ProbeGroup()
    for i in range(4):
        tetrode = generate_tetrode()
        tetrode.set_device_channel_indices(np.arange(4) + i * 4)
        probegroup.add_probe(tetrode)

    # set this to the recording
    recording_4_tetrodes = recording.set_probegroup(probegroup, group_mode='by_probe')
    # get group
    print(recording_4_tetrodes.get_channel_groups())
    # >>> [0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3]
    # similar to this
    print(recording_4_tetrodes.get_property('group'))
    # >>> [0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3]


**Option 1: Manual splitting**

.. code-block:: python

    # split into a dict
    recordings = recording_4_tetrodes.split_by(property='group', outputs='dict')
    print(recordings)

    # loop over recording and run a sorter
    # here the result is a dict of a sorting object
    sortings = {}
    for group, sub_recording in recordings.items():
        sorting = run_sorter('kilosort2', recording, output_folder=f"folder_KS2_group{group}")
        sortings[group] = sorting

**Option 2 : Automatic splitting**

.. code-block:: python

    # here the result is one sorting that agregate all sub sorting object
    aggregate_sorting = run_sorter_by_property('kilosort2', recording_4_tetrodes,
                                               grouping_property='group',
                                               working_folder='working_path')


Handling multi-segment recordings
---------------------------------

In several experiments, several acquisitions are performed in sequence, for example a 
baseline/intervention. In these cases, since the underlying spiking activity can be assumed to be
the same (or at least very similar), the recordings can be concatenated. This example shows how
to concatenate the recordings before spike sorting and how to split the sorted output based
on the concatenation.

Note that some sorters (tridesclous, spykingcircus2) handle a multi-segments paradigm directly. In
that case we will use the :py:func:`~spikeinterface.core.append_recordings()` function. Many sorters
do not handle multi-segment, and in that case we will use the
:py:func:`~spikeinterface.core.concatenate_recordings()` function.


.. code-block:: python


    # Let's create 4 recordings
    recordings_list = []
    for i in range(4):
      rec, _ = si.toy_example(duration=10., num_channels=4, seed=0, num_segments=1)
      recordings_list.append(rec)


    # Case 1: the sorter handles multi-segment objects

    multirecording = si.append_recordings(recordings_list)
    # let's set a probe
    multirecording = multirecording.set_probe(recording_single.get_probe())
    print(multirecording)
    # multirecording has 4 segments of 10s each

    # run tridesclous in multi-segment mode
    multisorting = si.run_sorter('tridesclous', multirecording)
    print(multisorting)

    # Case 2: the sorter DOES NOT handle multi-segment objects
    # The `concatenate_recordings()` mimics a mono-segment object that concatenates all segments
    multirecording = si.concatenate_recordings(recordings_list)
    # let's set a probe
    multirecording = multirecording.set_probe(recording_single.get_probe())
    print(multirecording)
    # multirecording has 1 segment of 40s each

    # run mountainsort4 in mono-segment mode
    multisorting = si.run_sorter('mountainsort4', multirecording)

See also the :ref:`multi_seg` section.


.. _compatible-sorters:

Supported Spike Sorters
-----------------------

Currently, we support many popular semi-automatic spike sorters.  Given the standardized, modular
design of our sorters, adding new ones is straightforward so we expect this list to grow in future
versions.


Here is the list of external sorters accessible using the run_sorter wrapper:

* **HerdingSpikes2** :code:`run_sorter('herdingspikes')`
* **IronClust** :code:`run_sorter('ironclust')`
* **Kilosort**  :code:`run_sorter('kilosort')`
* **Kilosort2** :code:`run_sorter('kilosort2')`
* **Kilosort2.5** :code:`run_sorter('kilosort2_5')`
* **Kilosort3** :code:`run_sorter('Kilosort3')`
* **PyKilosort** :code:`run_sorter('pykilosort')`
* **Klusta** :code:`run_sorter('klusta')`
* **Mountainsort4** :code:`run_sorter('mountainsort4')`
* **SpyKING Circus** :code:`run_sorter('spykingcircus')`
* **Tridesclous** :code:`run_sorter('tridesclous')`
* **Wave clus** :code:`run_sorter('waveclus')`
* **Combinato** :code:`run_sorter('combinato')`
* **HDSort** :code:`run_sorter('hdsort')`
* **YASS** :code:`run_sorter('yass')`


Here a list of internal sorter based on `spikeinterface.sortingcomponents`; they are totally
experimental for now:

* **Spyking circus2** :code:`run_sorter('spykingcircus2')`
* **Tridesclous2** :code:`run_sorter('tridesclous2')`

In 2023, we expect to add many more sorters to this list.


Installed Sorters
-----------------

To check which sorters are useable in a given Python environment, one can print the installed
sorters list. An example is shown in a pre-defined miniconda3 environment.


Then you can check the installed Sorter list,

.. code:: python

  from spikeinterface.sorters import installed_sorters
  installed_sorters()

which outputs,

.. parsed-literal::
  ['herdingspikes',
   'klusta',
   'mountainsort4',
   'spykingcircus',
   'tridesclous']


When trying to use an sorter that has not been installed in your environment, an installation
message will appear indicating how to install the given sorter,

.. code:: python

  recording = run_sorter('ironclust', recording)

throws the error,

.. parsed-literal::
  AssertionError: This sorter ironclust is not installed.
        Please install it with:

  To use IronClust run:

        >>> git clone https://github.com/jamesjun/ironclust
    and provide the installation path by setting the IRONCLUST_PATH
    environment variables or using IronClustSorter.set_ironclust_path().


Internal sorters
----------------

In 2022, we started the :py:mod:`spikeinterface.sortingcomponents` module to break into components a sorting pipeline.
Theses components can be gathered to create a new sorter. We already have 2 sorters to showcase this new module:

* :code:`spykingcircus2` (experimental, but ready to be tested) 
* :code:`tridesclous2` (experimental, not ready to be used)

There are some benefits of using these sorters:
  * they handle directly SpikeInterface objects, so they do not need any data copy.
  * they only require a few extra dependencies (like :code:`hdbscan`)


From the user's perspective, they behave exactly like "normal" sorters:

.. code-block:: python

    sorting = run_sorter("spykingcircus2", recording, "/tmp/folder")


Contributing
------------

There are 3 ways for contributing to the :py:mod:`spikeinterface.sorters` module:

  * helping in the containerization of spike sorters. This is managed on a separate GitHub repo,
    `spikeinterface-dockerfiles <https://github.com/SpikeInterface/spikeinterface-dockerfiles>`_. 
    If you find an error with a current container or would like to request a new spike sorter,
    please submit an Issue to this repo.
  * make a new wrapper of an existing external sorter.
  * make a new sorter based on :py:mod:`spikeinterface.sortingcomponents`





