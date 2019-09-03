Sorters installation
====================

An important aspect of spikeinterface is the `spikeinterface.sorters` module.
This module wrap several popular spike sorting tools.
This means that you can run multiple sorters on the same dataset without pain
through an unified layer with python code. See tutorial.

Wrapper also means that theses spike sorting tools **must be installed externally**.
Some of theses sorters are written in matlab, so you will also to install if you want
to use them (kilosort, kilosrt2, ironclust, jrclust, ...)
Some of then will also need some computing library  like cuda (kilosort, kilosort2),
opencl (tridesclous) to use hardware acceleration (GPU).

Here a list of implemented wrappers and some tips to install then on your local machine.
Installation tips are given for unbuntu platform and must transposed to other OS/distro.
We use **pip** but **conda** should also work in many cases.

If you experience installation problems please contact directly authors or mailing list
of theses tools. Please feel free to enhance this document.

Herdingspikes2
--------------

Url: https://github.com/mhhennig/hs2
Authors: Matthias Hennig, Jano Horvath,Cole Hurwitz, Oliver Muthmann, Albert Puente Encinas, Martino Sorbaro, Cesar Juarez Ramirez
Raimon Wintzer: GUI and visualisation
Installation::

    pip install herdingspikes

IronClust
---------

* Matlab based
* Url: https://github.com/jamesjun/ironclust
* Authors: James J. Jun
* Installation need matlab::

      git clone https://github.com/jamesjun/ironclust
      # provide installation path by setting the IRONCLUST_PATH environment variable
      # or using IronClustSorter.set_ironclust_path()

Kilosort
--------

* Matlab based, need cuda
* Url: https://github.com/cortex-lab/KiloSort
* Authors: Marius Pachitariu
* Installation need matlab and cudatoolkit::

      git clone https://github.com/cortex-lab/KiloSort
      # provide installation path by setting the KILOSORT_PATH environment variable
      # or using KilosortSorter.set_kilosort_path()

* See also for matlab/cuda: https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html
    
Kilosort2
---------

* Matlab based, need cuda
* Url: https://github.com/MouseLand/Kilosort2
* Authors: Marius Pachitariu
* Installation need matlab and cudatoolkit::

      git clone https://github.com/MouseLand/Kilosort2
      # provide installation path by setting the KILOSORT2_PATH environment variable
      # or using Kilosort2Sorter.set_kilosort2_path()

* See also for matlab/cuda: https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html


Klusta
------

* python based
* Url: https://github.com/kwikteam/klusta
* Authors: Cyrille Rossant, Shabnam Kadir, Dan Goodman, Max Hunter, Kenneth Harris
* Installation::

       pip install Cython h5py tqdm
       pip install click klusta klustakwik2

* See also: https://github.com/kwikteam/phy


Mountainsort4
-------------

* python based
* Url: https://github.com/flatironinstitute/mountainsort
* Authors: 	Jeremy Magland, Alex Barnett, Jason Chung, Loren Frank, Leslie Greengard
* Installation::

      pip install ml_ms4alg


SpykingCircus
-------------

* python based, need MPICH installed
* Url: https://spyking-circus.readthedocs.io
* Authors: Pierre Yger, Olivier Marre
* Installation::
      
        sudo apt install libmpich-dev
        pip install mpy4py
        pip install spyking-circus


tredesclous
-----------

* python based, run faster with opencl installed but optional
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

