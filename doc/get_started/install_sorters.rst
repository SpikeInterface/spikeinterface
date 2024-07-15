Installing Spike Sorters
========================


An important aspect of SpikeInterface is the :py:mod:`spikeinterface.sorters` module.
This module wraps many popular spike sorting tools, allowing you to run multiple sorters on the same dataset with
only a few lines of code and through Python.

Installing spike sorters can be painful! Many of them come with several requirements that could cause conflicts in
your Python environment. To make things easier, we have created Docker images for most of these sorters,
and in many cases the easiest way to run them is to do so via Docker or Singularity.
**This is the approach we recommend for all users.**
To run containerized sorters see our documentation here: :ref:`containerizedsorters`.

There are some cases where users will need to install the spike sorting algorithms in their own environment. If you
are on a system where it is infeasible to run Docker or Singularity containers, or if you are actively developing the
spike sorting software, you will likely need to install each spike sorter yourself.

Some of theses sorters are written in Matlab, so you will also need to install Matlab if you want
to use them (Kilosort, Kilosort2, Ironclust, ...).
Some of then will also need some computing libraries like CUDA (Kilosort, Kilosort2, Ironclust (optional)) or
opencl (Tridesclous) to use hardware acceleration (GPU).

Here is a list of the implemented wrappers and some instructions to install them on your local machine.
Installation instructions are given for an **Ubuntu** platform. Please check the documentation of the different spike
sorters to retrieve installation instructions for other operating systems.
We use **pip** to install packages, but **conda** should also work in many cases.

Some novel spike sorting algorithms are implemented directly in SpikeInterface using the
:py:mod:`spikeinterface.sortingcomponents` module. Checkout the :ref:`get_started/install_sorters:SpikeInterface-based spike sorters` section of this page
for more information!

If you experience installation problems please directly contact the authors of these tools or write on the
related mailing list, google group, GitHub issue page, etc.

Please feel free to enhance this document with more installation tips.

External sorters
----------------

Herdingspikes2
^^^^^^^^^^^^^^

* Python + C++
* Url: https://github.com/mhhennig/hs2
* Authors: Matthias Hennig, Jano Horvath,Cole Hurwitz, Oliver Muthmann, Albert Puente Encinas, Martino Sorbaro, Cesar Juarez Ramirez, Raimon Wintzer: GUI and visualisation
* Installation::

    pip install herdingspikes


HDSort
^^^^^^

* Matlab
* Url: https://git.bsse.ethz.ch/hima_public/HDsort.git
* Authors: Roland Diggelmann, Felix Franke
* Installation::

      git clone https://git.bsse.ethz.ch/hima_public/HDsort.git
      # provide installation path by setting the HDSORT_PATH environment variable
      # or using HDSortSorter.set_hdsort_path()


IronClust
^^^^^^^^^

* Matlab
* Url: https://github.com/jamesjun/ironclust
* Authors: James J. Jun
* Installation needs Matlab::

      git clone https://github.com/jamesjun/ironclust
      # provide installation path by setting the IRONCLUST_PATH environment variable
      # or using IronClustSorter.set_ironclust_path()


Kilosort
^^^^^^^^

* Matlab, requires CUDA
* Url: https://github.com/cortex-lab/KiloSort
* Authors: Marius Pachitariu
* Installation needs Matlab and cudatoolkit::

      git clone https://github.com/cortex-lab/KiloSort
      # provide installation path by setting the KILOSORT_PATH environment variable
      # or using KilosortSorter.set_kilosort_path()

* See also for Matlab/CUDA: https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html


Kilosort2
^^^^^^^^^

* Matlab, requires CUDA
* Url: https://github.com/MouseLand/Kilosort2
* Authors: Marius Pachitariu
* Installation needs Matlab and cudatoolkit::

      git clone https://github.com/MouseLand/Kilosort2
      # provide installation path by setting the KILOSORT2_PATH environment variable
      # or using Kilosort2Sorter.set_kilosort2_path()

* See also for Matlab/CUDA: https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html


Kilosort2.5
^^^^^^^^^^^

* Matlab, requires CUDA
* Url: https://github.com/MouseLand/Kilosort
* Authors: Marius Pachitariu
* Installation needs Matlab and cudatoolkit::

      git clone https://github.com/MouseLand/Kilosort
      # provide installation path by setting the KILOSORT2_5_PATH environment variable
      # or using Kilosort2_5Sorter.set_kilosort2_5_path()

* See also for Matlab/CUDA: https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html


Kilosort3
^^^^^^^^^

* Matlab, requires CUDA
* Url: https://github.com/MouseLand/Kilosort
* Authors: Marius Pachitariu
* Installation needs Matlab and cudatoolkit::

      git clone https://github.com/MouseLand/Kilosort
      # provide installation path by setting the KILOSORT3_PATH environment variable
      # or using Kilosort3Sorter.set_kilosort3_path()

* See also for Matlab/CUDA: https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html

Kilosort4
^^^^^^^^^

* Python, requires CUDA for GPU acceleration (highly recommended)
* Url: https://github.com/MouseLand/Kilosort
* Authors: Marius Pachitariu, Shashwat Sridhar, Carsen Stringer
* Installation::

      pip install kilosort==4.0 torch

* For more installation instruction refer to https://github.com/MouseLand/Kilosort


pyKilosort
^^^^^^^^^^

* Python, requires CUDA
* Url: https://github.com/int-brain-lab/pykilosort / https://github.com/MouseLand/pykilosort
* Authors: Marius Pachitariu, Shashwat Sridhar, Alexander Morley, Cyrille Rossant, Kush Bunga

* Install the python cuda toolkit. In principle, this should work::

    pip install cupy  (or pip install cupy-cudaXXX)

* However, conda installation could be less painful::

    conda install cupy

* Next, clone and install pykilosort. Note that we support the newer version on the `develop` branch and the `ibl_prod` version from the IBL fork::

    pip install phylib, pypandoc
    # recommended
    git clone --branch ibl_prod https://github.com/int-brain-lab/pykilosort
    # or
    git clone --branch develop https://github.com/MouseLand/pykilosort
    cd pykilosort
    pip install -r requirements.txt
    python setup.py install

* Alternatively, you can use the `pyks2.yml` environment file in the pykilosort repo and update your favorite environment with::

    conda env update --name my-fav-env --file pyks2.yml --prune

* See also https://github.com/MouseLand/pykilosort#installation


Mountainsort4
^^^^^^^^^^^^^

* Python
* Url: https://github.com/flatironinstitute/mountainsort
* Authors: 	Jeremy Magland, Alex Barnett, Jason Chung, Loren Frank, Leslie Greengard
* Installation::

      pip install mountainsort4

Mountainsort5
^^^^^^^^^^^^^

* Python
* Url: https://github.com/flatironinstitute/mountainsort5
* Authors: 	Jeremy Magland
* Installation::

      pip install mountainsort5

SpyKING CIRCUS
^^^^^^^^^^^^^^

* Python, requires MPICH
* Url: https://spyking-circus.readthedocs.io
* Authors: Pierre Yger, Olivier Marre
* Installation::

        sudo apt install libmpich-dev
        pip install mpi4py
        pip install spyking-circus --no-binary=mpi4py


Tridesclous
^^^^^^^^^^^

* Python, runs faster with opencl installed but optional
* Url: https://tridesclous.readthedocs.io
* Authors: Samuel Garcia, Christophe Pouzat
* Installation::

        pip install tridesclous

* Optional installation of opencl ICD and pyopencl for hardware acceleration::

        sudo apt-get install beignet (optional if Intel GPU)
        sudo apt-get install nvidia-opencl-XXX (optional if NVIDIA GPU)
        sudo apt-get install pocl-opencl-icd (optional for multi core CPU)
        sudo apt-get install opencl-headers ocl-icd-opencl-dev libclc-dev ocl-icd-libopencl1
        pip install pyopencl


Waveclus
^^^^^^^^

* Matlab
* Also supports Snippets (waveform cutouts) objects (:py:class:`~spikeinterface.core.BaseSnippets`)
* Url: https://github.com/csn-le/wave_clus/wiki
* Authors: Fernando Chaure, Hernan Rey and Rodrigo Quian Quiroga
* Installation needs Matlab::

      git clone https://github.com/csn-le/wave_clus/
      # provide installation path by setting the WAVECLUS_PATH environment variable
      # or using WaveClusSorter.set_waveclus_path()


Combinato
^^^^^^^^^

* Python
* Url: https://github.com/jniediek/combinato/wiki
* Authors: Johannes Niediek, Jan Boström, Christian E. Elger, Florian Mormann
* Installation::

      git clone https://github.com/jniediek/combinato
      # Then inside that folder, run:
      python setup_options.py
      # provide installation path by setting the COMBINATO_PATH environment variable
      # or using CombinatoSorter.set_combinato_path()

SpikeInterface-based spike sorters
----------------------------------

Thanks to the :py:mod:`spikeinterface.sortingcomponents` module, some spike sorting algorithms can now be fully implemented
with SpikeInterface.

SpykingCircus2
^^^^^^^^^^^^^^

This is a upgraded version of SpykingCircus, natively written in SpikeInterface.
The main differences are located in the clustering (now using on-the-fly features and less prone to finding
noise clusters), and in the template-matching procedure, which is now a fully orthogonal matching pursuit,
working not only at peak times but at all times, recovering more spikes close to noise thresholds.

* Python
* Requires: HDBSCAN and Numba
* Authors: Pierre Yger
* Installation::

        pip install hdbscan
        pip install spikeinterface
        pip install numba  (or conda install numba as recommended by conda authors)


Tridesclous2
^^^^^^^^^^^^

This is an upgraded version of Tridesclous, natively written in SpikeInterface.
#Same add his notes.

* Python
* Requires: HDBSCAN and Numba
* Authors: Samuel Garcia
* Installation::

      pip install hdbscan
      pip install spikeinterface
      pip install numba



Legacy Sorters
--------------

Klusta (LEGACY)
^^^^^^^^^^^^^^^

* Python
* Requires SpikeInterface<0.96.0 (and Python 3.7)
* Url: https://github.com/kwikteam/klusta
* Authors: Cyrille Rossant, Shabnam Kadir, Dan Goodman, Max Hunter, Kenneth Harris
* Installation::

       pip install Cython h5py tqdm
       pip install click klusta klustakwik2

* See also: https://github.com/kwikteam/phy


Yass (LEGACY)
^^^^^^^^^^^^^

* Python, CUDA, torch
* Requires SpikeInterface<0.96.0 (and Python 3.7)
* Url: https://github.com/paninski-lab/yass
* Authors: JinHyung Lee, Catalin Mitelut, Liam Paninski
* Installation::

      https://github.com/paninski-lab/yass/wiki/Installation-Local
