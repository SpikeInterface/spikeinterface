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

Running containerized spike sorters
-----------------------------------

The following code runs a containerized spike sorter:

```python
import spikeinterface.extractors as se
import spikeinterface.sorters as ss

test_recording, _ = se.toy_example(
    duration=30,
    seed=0,
    num_channels=64,
    num_segments=1
)

test_recording = test_recording.save(name='toy')

sorting = ss.run_kilosort3(recording=test_recording, output_folder="spykingcircus", singularity_image=True)
print(sorting)
```

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