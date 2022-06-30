.. _containerizedsorters:

Containerized Sorters
=====================

One of the biggest bottlenecks for users is installing spike sorting software. To alleviate this, we build and
maintain containerized versions of several popular spike sorters on the `SpikeInterface Docker Hub repository
<https://hub.docker.com/u/spikeinterface>`_.

The containerized approach has several advantages:
* Installation is much easier.
* Different spike sorters with conflicting dependencies can be easily run side-by-side.
* The results of the analysis are more reproducible and not dependant on the operating system
* MATLAB-based sorters can be run without a MATLAB licence.

The containers can be run in Docker or Singularity, so having Docker or Singularity installed is a prerequisite.

Running sorters in container docker/singularity
-----------------------------------------------

Some sorters are hard to install! To alleviate this headache, SI provides a built-in mechanism to run a spike sorting
job in a docker or singularity container.

We are maintaining a set of sorter-specific docker files in the `spikeinterface-dockerfiles repo <https://github.com/SpikeInterface/spikeinterface-dockerfiles>`_
and most of the docker images are available on Docker Hub from the `SpikeInterface organization <https://hub.docker.com/orgs/spikeinterface/repositories>`_.

singularity has an internal mechanism to convert docker images to singularity images.

singularity is often prefered because you don't need root privilege to run the container.
docker needs *almost  root* privilege

Running spike sorting in a docker container container just requires to:

1) have docker installed
2) have docker python SDK installed (:code:`pip install docker`)

or

1) have singularity installed
2) have `singularity python <https://singularityhub.github.io/singularity-cli/>`_ (:code:`pip install spython`)

When docker is installed, you can simply run the sorter in a specified docker image:

.. code-block:: python

    import spikeinterface.sorters as ss

    # recording is a RecordingExtractor object
    sorting_TDC = ss.run_tridesclous(
        recording,
        output_folder="tridesclous_output",
        docker_image="spikeinterface/tridesclous-base:1.6.4",
    )

And the same goes for singularity:

.. code-block:: python

    import spikeinterface.sorters as ss

    # recording is a RecordingExtractor object
    sorting_TDC = ss.run_tridesclous(
        recording,
        output_folder="tridesclous_output",
        singularity_image="spikeinterface/tridesclous-base:1.6.4",
    )


This will automatically check if the latest compiled kilosort3 docker image is present on your workstation and if it
is not the proper image will be downloaded from Docker Hub. The sorter will then run and output the results in the
designated folder. To run in Docker instead of Singularity, use `docker_image=True`. To use a specific image, set
either `docker_image` or `singularity_image` to a string, e.g.
`singularity_image="spikeinterface/kilosort3-compiled-base:0.1.0"`.


Contributing
------------
The containerization of the spike sorters is managed by a separate GitHub repo, `spikeinterface-dockerfiles
<https://github.com/SpikeInterface/spikeinterface-dockerfiles>`_. If you find an error with a current container
or would like to request a new spike sorter, please submit an Issue to this repo.