## Installation tips

You are not (yet) an expert in python installation. (conda vs pip, mananging environements, ...)

Here we propose a simple recipe to install spikeinterface and several sorter inside a anaconda environment for windows/mac user.


Step:

1. Download anaconda individual edition here https://www.anaconda.com/products/individual
2. Install it. Check the box “Add anaconda3 to my Path environment variable”. It make life easier for beginners.
3. Download the file "full_spikeinterface_environment.yml" from
   https://github.com/SpikeInterface/spikeinterface/tree/master/installation_tips
   and put it in "Documents" folder
4. Then open "anaconda powershell" (make a serach in your application).
5. Then run this: `conda env create --name si_env --file full_spikeinterface_environment.yml`


Then for before running a script you will need "select" this "environment" with `conda activate si_env`.


This environement will install:
 * spikeinterface full option
 * spikeinterface-gui
 * spyking-circus
 * tridesclous
 





