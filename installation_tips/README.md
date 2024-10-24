## Installation tips

If you are not (yet) an expert in Python installations, a main difficulty is choosing the installation procedure.
The main ideas you need to know before starting:
 * python itself can be distributed and installed many many ways.
 * python itself do not contain so many features for scientifique computing you need to install "packages".
   Numpy, scipy, matplotlib, spikeinterface, ... are python packages that have a complicated dependency graph between then. "uv"
 * installing package  can be distributed and installed several ways (pip, conda, uv, mamba, ...)
 * installing many packages at once is challenging (because of the depenency graph) so you need to do it in an "isolated environement"
   to not destroy any previous installation. You need to see an "environement" as a sub installtion in dedicated folder.

Choosing the installator + a environement manager + a package installer is a nightmare for beginners.
The main options are:
  * use "anaconda" that do everything. The most popular but bad idea because : ultra slow and agressive licensing (not free anymore)
  * use python from the system or python.org + venv + pip : good idea for linux users.
  * use "uv" : a new, fast and simple. We recommand this for beginners on evry os.

Here we propose a steps by step recipe for beginers based on "uv".
We used to propose here a solution based on anaconda. It is kept here for a while but we do not recommand it anymore.


This environment will install:
 * spikeinterface full option
 * spikeinterface-gui
 * kilosort4

Kilosort, Ironclust and HDSort are MATLAB based and need to be installed from source.

### Quick installation

1. On macOS and Linux. Open a terminal and do
   `$ curl -LsSf https://astral.sh/uv/install.sh | sh`
1. On windows. Open a powershell and do
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. exit session and log again.
3. Download with right click + save this file corresponding in "Documents" folder:
    * [`requirements_stable.txt`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/requirements_stable.txt)
4. open terminal or powershell
5. `uv venv si_env --python 3.11`
6. `source `source si_env/bin/activate` (you should have `(si_env)` in your terminal)
7. `uv pip install -r Documents/requirements_stable.txt`


More detail on [uv here](https://github.com/astral-sh/uv).

## Installing before release

Some tools in the spikeinteface ecosystem are getting regular bug fixes (spikeinterface, spikeinterface-gui, probeinterface, python-neo, sortingview).
We are making releases 2 to 4 times a year. In between releases if you want to install from source you can use the `requirements_rolling.txt` file to create the environment. This will install the packages of the ecosystem from source.
This is a good way to test if patch fix your issue.


### Check the installation


If you want to test the spikeinterface install you can:

1. Download with right click + save the file [`check_your_install.py`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/check_your_install.py)
    and put it into the "Documents" folder

2. Open the Anaconda Command Prompt (Windows) or Terminal (Mac)
3. If not in your "Documents" folder type `cd Documents`
4. Run this:
    ```
    conda activate si_env
    python check_your_install.py
    ```
5. If a windows user to clean-up you will also need to right click + save [`cleanup_for_windows.py`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/cleanup_for_windows.py)
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

### Legacy installation using anaconda (not recomanded)

Steps:

1. Download anaconda individual edition [here](https://www.anaconda.com/download)
2. Run the installer. Check the box “Add anaconda3 to my Path environment variable”. It makes life easier for beginners.
3. Download with right click + save the file corresponding to your OS, and put it in "Documents" folder
    * [`full_spikeinterface_environment_windows.yml`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/full_spikeinterface_environment_windows.yml)
    * [`full_spikeinterface_environment_mac.yml`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/full_spikeinterface_environment_mac.yml)
4. Then open the "Anaconda Command Prompt" (if Windows, search in your applications) or the Terminal (for Mac users)
5. If not in the "Documents" folder type `cd Documents`
6. Then run this depending on your OS:
    * `conda env create --file full_spikeinterface_environment_windows.yml`
    * `conda env create --file full_spikeinterface_environment_mac.yml`


Done! Before running a spikeinterface script you will need to "select" this "environment" with `conda activate si_env`.

Note for **linux** users : this conda recipe should work but we recommend strongly to use **pip + virtualenv**.




## Installing before release

Some tools in the spikeinteface ecosystem are getting regular bug fixes (spikeinterface, spikeinterface-gui, probeinterface, python-neo, sortingview).
We are making releases 2 to 4 times a year. In between releases if you want to install from source you can use the `full_spikeinterface_environment_rolling_updates.yml` file to create the environment. This will install the packages of the ecosystem from source.
This is a good way to test if patch fix your issue.
