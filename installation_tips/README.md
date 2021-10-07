## Installation tips

You are not (yet) an expert in python installation. (conda vs pip, mananging environements, ...)

Here we propose a simple recipe to install spikeinterface and several sorter inside a anaconda environment for windows/mac user.

This environement will install:
 * spikeinterface full option
 * spikeinterface-gui
 * spyking-circus
 * tridesclous


### Quick installation

Steps:

1. Download anaconda individual edition here https://www.anaconda.com/products/individual
2. Install it. Check the box “Add anaconda3 to my Path environment variable”. It make life easier for beginners.
3. Download with right click + save the file "full_spikeinterface_environment.yml"
    https://raw.githubusercontent.com/SpikeInterface/spikeinterface/master/installation_tips/full_spikeinterface_environment.yml
   and put it in "Documents" folder
4. Then open "anaconda powershell" (make a serach in your application).
5. Then run this: `conda env create --file full_spikeinterface_environment.yml`


Then for before running a script you will need "select" this "environment" with `conda activate si_env`.


### Check it


If you want a first try you can:

1. Download with right click + save the file "full_spikeinterface_environment.yml"
    https://raw.githubusercontent.com/SpikeInterface/spikeinterface/master/installation_tips/check_your_install.py
    and put it in "Documents" folder

2. open anaconda powershell
3. Run this
    ```
    cd Documents
    conda activate si_env
    python check_your_install.py
    ```

This script:
  * try to import spikeinterface
  * run tridesclous
  * run spyking-circus
  * open spikeinterface-gui





