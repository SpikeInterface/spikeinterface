.. _containerizedsorters:

Containerized Sorters
=====================

One of the biggest bottlenecks for users is installing spike sorting software. To alleviate this, we build and
maintain containerized versions for most spike sorters.

The containerized approach has several advantages:  

* Installation is much easier.  
* Different spike sorters with conflicting dependencies can be easily run side-by-side.  
* The results of the analysis are more reproducible and not dependant on the operating system  
* MATLAB-based sorters can be run without a MATLAB licence.  

The containers can be run in Docker or Singularity, so having Docker or Singularity installed is a prerequisite.

Running sorters in container docker/singularity
-----------------------------------------------

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

Contributing
------------

The containerization of spike sorters is managed on a separate GitHub repo, `spikeinterface-dockerfiles
<https://github.com/SpikeInterface/spikeinterface-dockerfiles>`_. 
If you find an error with a current container or would like to request a new spike sorter, please submit an Issue to this repo.