Installing Spike Sorters 
========================

An important aspect of spikeinterface is the `spikeinterface.sorters` module.
This module wraps many popular spike sorting tools.
This means that you can run multiple sorters on the same dataset with only a few lines of code
and through Python. 

These spike sorting algorithms **must be installed externally**.
Some of theses sorters are written in Matlab, so you will also to install Matlab if you want
to use them (Kilosort, Kilosort2, Ironclust, JRclust, ...)
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
* Authors: Matthias Hennig, Jano Horvath,Cole Hurwitz, Oliver Muthmann, Albert Puente Encinas, Martino Sorbaro, Cesar Juarez Ramirez
           Raimon Wintzer: GUI and visualisation
* Installation::

    pip install herdingspikes

IronClust
---------

* Matlab based
* Url: https://github.com/jamesjun/ironclust
* Authors: James J. Jun
* Installation need Matlab::

      git clone https://github.com/jamesjun/ironclust
      # provide installation path by setting the IRONCLUST_PATH environment variable
      # or using IronClustSorter.set_ironclust_path()

Kilosort
--------

* Matlab based, needs cuda
* Url: https://github.com/cortex-lab/KiloSort
* Authors: Marius Pachitariu
* Installation needs Matlab and cudatoolkit::

      git clone https://github.com/cortex-lab/KiloSort
      # provide installation path by setting the KILOSORT_PATH environment variable
      # or using KilosortSorter.set_kilosort_path()

* See also for Matlab/cuda: https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html
    
Kilosort2
---------

* Matlab based, needs cuda
* Url: https://github.com/MouseLand/Kilosort2
* Authors: Marius Pachitariu
* Installation need Matlab and cudatoolkit::

      git clone https://github.com/MouseLand/Kilosort2
      # provide installation path by setting the KILOSORT2_PATH environment variable
      # or using Kilosort2Sorter.set_kilosort2_path()

* See also for Matlab/cuda: https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html


Klusta
------

* Python based
* Url: https://github.com/kwikteam/klusta
* Authors: Cyrille Rossant, Shabnam Kadir, Dan Goodman, Max Hunter, Kenneth Harris
* Installation::

       pip install Cython h5py tqdm
       pip install click klusta klustakwik2

* See also: https://github.com/kwikteam/phy


Mountainsort4
-------------

* Python based
* Url: https://github.com/flatironinstitute/mountainsort
* Authors: 	Jeremy Magland, Alex Barnett, Jason Chung, Loren Frank, Leslie Greengard
* Installation::

      pip install ml_ms4alg


SpykingCircus
-------------

* Python based, needs MPICH installed
* Url: https://spyking-circus.readthedocs.io
* Authors: Pierre Yger, Olivier Marre
* Installation::
      
        sudo apt install libmpich-dev
        pip install mpy4py
        pip install spyking-circus --no-binary=mpi4py


Tridesclous
-----------

* Python based, runs faster with opencl installed but optional
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

* Matlab based
* Url: https://github.com/csn-le/wave_clus/wiki
* Authors: Fernando Chaure, Hernan Rey and Rodrigo Quian Quiroga
* Installation needs Matlab::

      git clone https://github.com/csn-le/wave_clus/
      # provide installation path by setting the WAVECLUS_PATH environment variable
      # or using WaveClusSorter.set_waveclus_path()
