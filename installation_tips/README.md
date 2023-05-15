## Installation tips

If you are not (yet) an expert in Python installations (conda vs pip, mananging environements, etc.), 
here we propose a simple recipe to install `spikeinterface` and several sorters inside an anaconda 
environment for windows/mac users.

This environment will install:
 * spikeinterface full option
 * spikeinterface-gui
 * phy
 * tridesclous
 * spyking-circus (not on mac)
 * herdingspikes (not on windows)

Kilosort, Ironclust and HDSort are MATLAB based and need to be installed from source.
Klusta does not work anymore with python3.8 you should create a similar environment with python3.6.

### Quick installation

Steps:

1. Download anaconda individual edition [here](https://www.anaconda.com/products/individual)
2. Run the installer. Check the box “Add anaconda3 to my Path environment variable”. It makes life easier for beginners.
3. Download with right click + save the file corresponding to your OS, and put it in "Documents" folder
    * [`full_spikeinterface_environment_windows.yml`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/master/installation_tips/full_spikeinterface_environment_windows.yml)
    * [`full_spikeinterface_environment_mac.yml`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/master/installation_tips/full_spikeinterface_environment_mac.yml)
4. Then open the "Anaconda Command Prompt" (if Windows, search in your applications) or the Terminal (for Mac users)
5. If not in the "Documents" folder type `cd Documents`
6. Then run this depending on your OS:
    * `conda env create --file full_spikeinterface_environment_windows.yml`
    * `conda env create --file full_spikeinterface_environment_mac.yml`


Done! Before running a spikeinterface script you will need to "select" this "environment" with `conda activate si_env`.

Note for **linux** users : this conda recipe should work but we recommend strongly to use **pip + virtualenv**.


### Check the installation


If you want to test the spikeinterface install you can:

1. Download with right click + save the file [`check_your_install.py`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/master/installation_tips/check_your_install.py)
    and put it into the "Documents" folder

2. Open the Anaconda Command Prompt (Windows) or Terminal (Mac)
3. If not in your "Documents" folder type `cd Documents`
4. Run this:
    ```
    conda activate si_env
    python check_your_install.py
    ```
5. If a windows user to clean-up you will also need to right click + save [`cleanup_for_windows.py`](https://raw.githubusercontent.com/SpikeInterfacemaster/installation_tips/cleanup_for_windows.py)
Then transfer `cleanup_for_windows.py` into your "Documents" folder. Finally run :
   ```
   python cleanup_for_windows.py
   ```
   
This script tests the following:
  * importing spikeinterface
  * running tridesclous
  * running spyking-circus (not on mac)
  * running herdingspikes (not on windows)
  * opening the spikeinterface-gui
  * exporting to Phy

