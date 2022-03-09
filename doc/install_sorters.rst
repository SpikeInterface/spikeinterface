Installing Spike Sorters
========================

An important aspect of spikeinterface is the `spikeinterface.sorters` module.
This module wraps many popular spike sorting tools.
This means that you can run multiple sorters on the same dataset with only a few lines of code
and through Python.

These spike sorting algorithms **must be installed externally**.
Some of theses sorters are written in Matlab, so you will also to install Matlab if you want
to use them (Kilosort, Kilosort2, Ironclust, ...)
Some of then will also need some computing library like CUDA (Kilosort, Kilosort2, Ironclust (optional)) or
opencl (Tridesclous) to use hardware acceleration (GPU).

Here is a list of the implemented wrappers and some instructions to install them on your local machine.
Installation instructions are given for an **Unbuntu** platform. Please check the documentation of the different spike
sorters to retrieve installation instructions for other operating systems.
We use **pip** to install packages, but **conda** should also work in many cases.

If you experience installation problems please directly contact the authors of theses tools or write on the
related mailing list, google group, etc.

Please feel free to enhance this document with more installation tips.

Herdingspikes2
--------------

* Python + C++
* Url: https://github.com/mhhennig/hs2
* Authors: Matthias Hennig, Jano Horvath,Cole Hurwitz, Oliver Muthmann, Albert Puente Encinas, Martino Sorbaro, Cesar Juarez Ramirez, Raimon Wintzer: GUI and visualisation
* Installation::

    pip install herdingspikes

HDSort
-------

* Matlab
* Url: https://git.bsse.ethz.ch/hima_public/HDsort.git
* Authors: Roland Diggelmann, Felix Franke
* Installation::

      git clone https://git.bsse.ethz.ch/hima_public/HDsort.git
      # provide installation path by setting the HDSORT_PATH environment variable
      # or using HDSortSorter.set_hdsort_path()

IronClust
---------

* Matlab
* Url: https://github.com/jamesjun/ironclust
* Authors: James J. Jun
* Installation need Matlab::

      git clone https://github.com/jamesjun/ironclust
      # provide installation path by setting the IRONCLUST_PATH environment variable
      # or using IronClustSorter.set_ironclust_path()

Kilosort
--------

* Matlab, requires CUDA
* Url: https://github.com/cortex-lab/KiloSort
* Authors: Marius Pachitariu
* Installation needs Matlab and cudatoolkit::

      git clone https://github.com/cortex-lab/KiloSort
      # provide installation path by setting the KILOSORT_PATH environment variable
      # or using KilosortSorter.set_kilosort_path()

* See also for Matlab/cuda: https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html

Kilosort2
---------

* Matlab, requires CUDA
* Url: https://github.com/MouseLand/Kilosort2
* Authors: Marius Pachitariu
* Installation needs Matlab and cudatoolkit::

      git clone https://github.com/MouseLand/Kilosort2
      # provide installation path by setting the KILOSORT2_PATH environment variable
      # or using Kilosort2Sorter.set_kilosort2_path()

* See also for Matlab/cuda: https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html


Kilosort2.5
-----------

* Matlab, requires CUDA
* Url: https://github.com/MouseLand/Kilosort
* Authors: Marius Pachitariu
* Installation needs Matlab and cudatoolkit::

      git clone https://github.com/MouseLand/Kilosort
      # provide installation path by setting the KILOSORT2_5_PATH environment variable
      # or using Kilosort2_5Sorter.set_kilosort2_path()

* See also for Matlab/cuda: https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html

Kilosort3
-----------

* Matlab, requires CUDA
* Url: https://github.com/MouseLand/Kilosort
* Authors: Marius Pachitariu
* Installation needs Matlab and cudatoolkit::

      git clone https://github.com/MouseLand/Kilosort
      # provide installation path by setting the KILOSORT3_PATH environment variable
      # or using Kilosort3Sorter.set_kilosort3_path()

* See also for Matlab/cuda: https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html


pykilosort
----------

* python, requires cuda
* Url: https://github.com/MouseLand/pykilosort
* Authors: Marius Pachitariu, Shashwat Sridhar, Alexander Morley, Cyril Rossant

* Installation needs Matlab and cudatoolkit::

    pip install cupy  (or pip install cupy-cudaXXX)  >>> can be quite hard
    pip install phylib, pypandic
    git clone https://github.com/MouseLand/pykilosort
    cd pykilosort
    python setup.py install

* See also https://github.com/MouseLand/pykilosort#installation


Klusta
------

* Python
* Url: https://github.com/kwikteam/klusta
* Authors: Cyrille Rossant, Shabnam Kadir, Dan Goodman, Max Hunter, Kenneth Harris
* Installation::

       pip install Cython h5py tqdm
       pip install click klusta klustakwik2

* See also: https://github.com/kwikteam/phy


Mountainsort4
-------------

* Python
* Url: https://github.com/flatironinstitute/mountainsort
* Authors: 	Jeremy Magland, Alex Barnett, Jason Chung, Loren Frank, Leslie Greengard
* Installation::

      pip install mountainsort4


SpykingCircus
-------------

* Python, requires MPICH
* Url: https://spyking-circus.readthedocs.io
* Authors: Pierre Yger, Olivier Marre
* Installation::

        sudo apt install libmpich-dev
        pip install mpi4py
        pip install spyking-circus --no-binary=mpi4py


Tridesclous
-----------

* Python, runs faster with opencl installed but optional
* Url: https://tridesclous.readthedocs.io
* Authors: Samuel Garcia, Christophe Pouzat
* Installation::

        pip install tridesclous

* Optional installation of opencl ICD and pyopencl for hardware acceleration::

        sudo apt-get install beignet (optional if intel GPU)
        sudo apt-get install nvidia-opencl-XXX (optional if nvidia GPU)
        sudo apt-get install pocl-opencl-icd (optional for multi core CPU)
        sudo apt-get install opencl-headers ocl-icd-opencl-dev libclc-dev ocl-icd-libopencl1
        pip install pyopencl

Waveclus
--------

* Matlab
* Url: https://github.com/csn-le/wave_clus/wiki
* Authors: Fernando Chaure, Hernan Rey and Rodrigo Quian Quiroga
* Installation needs Matlab::

      git clone https://github.com/csn-le/wave_clus/
      # provide installation path by setting the WAVECLUS_PATH environment variable
      # or using WaveClusSorter.set_waveclus_path()


Combinato
---------

* Python
* Url: https://github.com/jniediek/combinato/wiki
* Authors: Johannes Niediek, Jan Boström, Christian E. Elger, Florian Mormann
* Installation::

      git clone https://github.com/jniediek/combinato
      # Then inside that folder, run:
      python setup_options.py
      # provide installation path by setting the COMBINATO_PATH environment variable
      # or using CombinatoSorter.set_combinato_path()

Yass
----

* Python, cuda, torch
* Url: https://github.com/paninski-lab/yass
* Authors: Liam Paninski
* Installation::

      https://github.com/paninski-lab/yass/wiki/Installation-Local
