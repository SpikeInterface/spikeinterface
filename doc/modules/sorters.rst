Sorters module
==============

The :py:mod:`spikeinterface.sorters` module is where spike sorting happens!

TODO : exaplain external vs internal

On one hands, SpikeInterface provides wrapper classes to many commonly used spike sorters like
kilosort, spkykingcirucs (see :ref:`compatible-sorters`) All theses sorter classes inherit
from the :py:class:`~spikeinterface.sorters.BaseSorter` class, which provides the common tools to
run spike sorters.

On the other hand spikeinterface have internally some experimental sorters (**spkykingcicurs2**) 
that do not depend on external tools but depend on the `spikeinterface.sortingcomponents` layer.

A drawback using external sorters is the installation of theses tools. Sometimes they need matlab, 
specific version of cuda, specific gcc version very or even worst : unmaintained version of
python/numpy. In that case, spikeinterface offer the mechanism of running external sorters inside a
container (docker/singularity) with the sorter pre-installed. See :ref:`containerizedsorters`.


External sorter : wrappers concept
----------------------------------

When running external sorters, we use the concept of "wrappers". In short we have some python code
that generate external code (for instance matlab) and also external config files. Then the generated
code is run in the background. When finish the output is retrieve back into python and
the spikeinterface object :py:func:`~spiekinterface.core.BaseSorting`.

For instance, we have internally a class `Kilosort2_5Sorter` that handle:
  * Formating the data and parameters for kilosort2.5, using `Kilosort2_5Sorter.setup_recording()`
  * Running matlab and kilosort2.5 code in the folder, using `Kilosort2_5Sorter.run_from_folder()`
  * Retriving the results using `Kilosort2_5Sorter.get_result_from_folder()`

From the user point of view this is totally hidden and we just need to do and this process
is run with:

.. code-block:: python

    sorting = ss.run_sorter("kilosort2_5", recording, output_folder="/path/to/working_folder_ks2.5")


Example
-------

So running and trying sorters is easy as this

The :code:`sorters` includes :py:ref:`~spikeinterface.sorters.run_sorter()` functions
to easily run spike sorters:

.. code-block:: python

    recording = read_openephys(...)

    from spikeinterface.sorters import run_sorter

    # run tridesclous
    sorting_TDC = run_sorter("tridesclous", recording, output_folder="/folder_TDC")
    # run kilosort.5
    sorting_KS2_5 = run_sorter("kilosort2_5", recording, output_folder="/folder_KS2.5")
    # run IronClust
    sorting_IC = run_sorter("ironclust", recording, output_folder="/folder_ironclust")
    # run pykilosort
    sorting_pyKS = run_sorter("pykilosort", recording, output_folder="/folder_pyks")
    # run spykingcircus
    sorting_SC = run_sorter("ironclust", recording, output_folder="/folder_SC")


Then the output, which is a sorting object can be saved easily or directly post processed.

.. code-block:: python

    sorting_TDC.save(folder='/path/to/my_spiketrains_with_tridesclous')


:py:ref:`~spikeinterface.sorters.run_sorter()` have options :

  * to remove or not the working folder (`output_folder`) 
    with :code:`remove_existing_folder=True/False`, this same lot of space because some sorters
    need data duplication!
  * to control ther verbosity :code:`verbose=False/True`
  * raise error or not :code:`raise_error=False/True`

Parameters from sorters can be controlled directly in the `run_sorter()` function:

.. code-block:: python

    sorting_TDC = run_sorter('tridesclous', recording, output_folder="/folder_TDC",
                                detect_threshold=8.)

    sorting_KS2_5 = run_sorter("kilosort2_5", recording, output_folder="/folder_KS2.5"
                               do_correction=false, preclust_threshold=6, freq_min=200.)


Parameters from all classes can be listed with theses functions:

.. code-block:: python

    params = get_default_sorter_params('spykingcircus')
    print(params)

    desc = get_sorter_params_description('spykingcircus')
    print(desc)

.. parsed-literal::

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



Run several sorters in parallel
-------------------------------

The :py:mod:`spikeinterface.sorters` includes also includes tools to run several spike sorting jobs sequentially or
in parallel. This can be done with the :py:func:`spikeinterface.sorters.run_sorters()` function by specifying
an :code:`engine` that supports parallel processing joblib or slurm).

.. code-block:: python

    recording, sorting_true = toy_example(duration=10, seed=0, num_segments=1)
    print(recording)
    # cache this recording to make it "dumpable"
    recording = recording.save(name='toy')
    print(recording)

    recordings = {'rec1' : recording, 'rec2': another_recording}
    sorter_list = ['herdingspikes', 'tridesclous']
    sorter_params = { 'herdingspikes': {'clustering_bandwidth' : 8},
                      'tridesclous': {'detect_threshold' : 5.},
                    }
    sorting_output = run_sorters(sorter_list, recordings, working_folder='tmp_some_sorters', 
                                    mode_if_folder_exists='overwrite', sorter_params=sorter_params)

    # the output is a dict with 2 keys
    for (rec_name, sorter_name), sorting in sorting_output.items():
        print(rec_name, sorter_name, ':', sorting.get_unit_ids())

After the jobs are run, the :code:`sorting_outputs` is a dictionary with :code:`(rec_name, sorter_name)` as key (e.g.
:code:`('rec1', 'tridesclous')` in this example), and the corresponding :code:`SortingExtractor` as value.



:py:func:`~spikeinterface.sorters.run_sorters()` have several "engine" to launch the computation
sequentially (loop) or in parralel (with joblib) or inside a job manager (slurm).

.. code-block:: python

  run_sorters(sorter_list, recordings, engine='loop')

  run_sorters(sorter_list, recordings, engine='joblib', engine_kwargs={'n_jobs': 2})

  run_sorters(sorter_list, recordings, engine='slurm',
              engine_kwargs={'cpus_per_task': 10, 'mem', '5G'})


Sorting by groups
-----------------

Sometimes you may want to spike sort using specific grouping : for instance sorting by tetrode
groups or if the probe is multi shanks sorting by shank or if the recording itself have several
probes sorting by probe. Alternatively, for long silicon probes, such as Neuropixels, you could
sort different areas separately, for example hippocampus and thalamus. This is a very common need.

Internally, the recording object have the ability to split (lasily) into a dict of sub-recordings.
So it is easy to loop over this dict to run sequentially sorting on theses sub recordings (aka 
ChannelSliceRecording).

spikeinterface also propose a high level function to automatise the process of splitting the
recording and then aggregating the result: :py:func:`~spikeinterface.sorters.run_sorter_by_property()`

This create 16 channels recording with 4 tetrodes.

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




**Example 1 : manual split**

.. code-block:: python

    # split into a dict
    recordings = recording_4_tetrodes.split_by(property='group', outputs='dict')
    print(recordings)

    # loop over recording and run a sorter
    # here the result is a dict a sorting object
    sortings = {}
    for group, sub_recording in recordings.items():
        sorting = run_sorter('kilosort2', recording, output_folder=f"/folder_KS2_group{group}")
        sortings[group] = sorting

**Example 2 : automatic split**

.. code-block:: python

    # here the result is one sorting that agregate all sub sorting object
    aggregate_sorting = run_sorter_by_property('kilosort2', recording_4_tetrodes,
                                             grouping_property='group', working_folder='/work_path')


Working on recording with multiple segments
-------------------------------------------

In several experiments, several recordings are performed in sequence, for example a 
baseline/intervention. In these cases, since the underlying spiking activity can be assumed to be
the same (or at least very similar), the recordings can be concatenated. This example shows how
to concatenate the recordings before spike sorting and how to split the sorted output based
on the concatenation.

Note that some sorters (tridesclous, spykingcircus2) handle directly multi segments paradigm, in
that case we will use the :py:func:`~spikeinterface.core.append_recordings()` function. Many sorters
do not handle multi segment, in that case we will use the
:py:func:`~spikeinterface.core.concatenate_recordings()` function.


.. code-block:: python


    # Let's create 4 recordings
    recordings_list = []
    for i in range(4):
      rec, _ = si.toy_example(duration=10., num_channels=4, seed=0, num_segments=1)
      recordings_list.append(rec)


    # Case 1. : the sorter handle multi segment

    multirecording = si.append_recordings(recordings_list)
    # lets put a probe
    multirecording = multirecording.set_probe(recording_single.get_probe())
    print(multirecording)
    # multi recording have 4 segments of 10s each

    # run tridesclous in multi segment mode
    multisorting = si.run_sorter('tridesclous', multirecording)
    print(multisorting)

    # Case 2. : the sorter DO NOT handle multi segment
    # In that case the `concatenate_recordings()` mimic a mono segment that concatenate all segment

    multirecording = si.concatenate_recordings(recordings_list)
    # lets put a probe
    multirecording = multirecording.set_probe(recording_single.get_probe())
    print(multirecording)
    # multi recording have 1 segment of 40s each

    # run klusta in mono segment mode
    multisorting = si.run_sorter('klusta', multirecording)

See also :ref:`multi_seg`


.. _compatible-sorters:

Supported Spike Sorters
-----------------------

Currently, we support many popular semi-automatic spike sorters.  Given the standardized, modular
design of our sorters, adding new ones is straightforward so we expect this list to grow in future
versions.


Here the list of external sorters using wrapper:

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
* **yass** :code:`run_sorter('yass')`

Here a list of internal sorter based on `spiekinterface.sortingcomponents`, they are totally
experimentals for now:

* **Spyking circus2** :code:`run_sorter('spykingcircus2')`
* **tridesclous2** :code:`run_sorter('tridesclous2')`

In 2023, we expect to have more sorters in that list.



Installed Sorters
-----------------

To check which sorters are useable in a given python environment, one can print the installed
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


Running spike sorting in a docker container container just requires to:

1) have docker installed
2) have docker python SDK installed (:code:`pip install docker`)

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
For :code:`Docker Engine`, you also need to enable Docker to run without :code:`sudo` priviledges 
following `this post-install guide <https://docs.docker.com/engine/install/linux-postinstall/>`_

The containers are built with Docker, but Singularity has an internal mechanism to convert docker images.
Using Singularity is often prefered due to its simpler approach with regard to root privilege.

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



Internal sorters
----------------

In 2022, we started the `spikeinterface.sortingcomponents` module to break into components a sorting pipeline.
Theses components can be gather to create a new sorter. We have already 2 sorters to show case this new module:
:code:`spykingcircus2` (experimental but ready to use) and :code:`tridesclous2` (very very experimental not to used)

There are some benefit of using theses sorter:
  * theye handle directly spikeinterface recording object, so no data copy.
  * need few extra dependencies (like `hdbscan`)


From the end user, they behave exactly the same:

.. code-block:: python


    sorting = run_sorter("spykingcircus2", recording, "/tmp/folder")



Contributing
------------

There are 3 ways for contributing to `spiekinterface.sorters`:

  * helping in the containerization of spike sorters. This is managed on a separate GitHub repo,
    `spikeinterface-dockerfiles <https://github.com/SpikeInterface/spikeinterface-dockerfiles>`_. 
    If you find an error with a current container or would like to request a new spike sorter,
    please submit an Issue to this repo.
  * make a new wrapper of external existing sorter.
  * make a new sorter based on `spikeinterface.sortingcomponents`





