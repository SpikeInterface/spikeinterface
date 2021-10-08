## Installation tips

If you are not (yet) an expert in Python installations (conda vs pip, mananging environements, etc.), 
here we propose a simple recipe to install :code:`spikeinterface` and several sorters inside a anaconda 
environment for windows/mac user.

This environment will install:
 * spikeinterface full option
 * spikeinterface-gui
 * phy
 * spyking-circus
 * tridesclous


### Quick installation

Steps:

1. Download anaconda individual edition [here](https://www.anaconda.com/products/individual)
2. Run the installer. Check the box “Add anaconda3 to my Path environment variable”. It makes life easier for beginners.
3. Download with right click + save the file [`full_spikeinterface_environment_*.yml`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/master/installation_tips/full_spikeinterface_environment.yml)
    
   and put it in "Documents" folder (choose the `windows` or `linux-mac` file depending on your operating system)
4. Then open the "Anaconda Command Prompt" (search in your applications)
5. Then run this: `conda env create --file full_spikeinterface_environment_*.yml`


Done! Before running a spikeinterface script you will need "select" this "environment" with `conda activate si_env`.


### Check the installation


If you want a first try you can:

1. Download with right click + save the file [`check_your_install.py`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/master/installation_tips/check_your_install.py)
    and put it in "Documents" folder

2. Open the Anaconda Command Prompt
3. Run this:
    ```
    cd Documents
    conda activate si_env
    python check_your_install.py
    ```

This script tests the following:
  * import spikeinterface
  * run tridesclous
  * run spyking-circus
  * open spikeinterface-gui
  * export to Phy
  * run Phy





