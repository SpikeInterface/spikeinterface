## Installation tips

If you are not (yet) an expert in Python installations (conda vs pip, mananging environements, etc.), 
here we propose a simple recipe to install `spikeinterface` and several sorters inside a anaconda 
environment for windows/mac user.

This environment will install:
 * spikeinterface full option
 * spikeinterface-gui
 * phy
 * tridesclous
 * spyking-circus (not on mac)
 * herdinspikes (not on windows)

Kilosort, Ironclust and HDSort are MATLAB based and need to be installed by source.
Klusta does not work anymore with python3.8 you should create a similar environment with python3.6

### Quick installation

Steps:

1. Download anaconda individual edition [here](https://www.anaconda.com/products/individual)
2. Run the installer. Check the box “Add anaconda3 to my Path environment variable”. It makes life easier for beginners.
3. Download with right click + save the file corresponding to your OS, and put it in "Documents" folder
    * [`full_spikeinterface_environment_windows.yml`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/master/installation_tips/full_spikeinterface_environment_windows.yml)
    * [`full_spikeinterface_environment_mac.yml`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/master/installation_tips/full_spikeinterface_environment_mac.yml)
4. Then open the "Anaconda Command Prompt" (search in your applications)
5. Then run this depending your OS:
    * `conda env create --file full_spikeinterface_environment_windows.yml`
    * `conda env create --file full_spikeinterface_environment_mac.yml`


Done! Before running a spikeinterface script you will need "select" this "environment" with `conda activate si_env`.

Note for **linux** users : this conda recipe should work but we recommand strongly to use **pip + virtualenv**.


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
  * run spyking-circus (not on mac)
  * run herdinspikes (not on windows)
  * open spikeinterface-gui
  * export to Phy
  * run Phy
