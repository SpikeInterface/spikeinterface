## Installation tips

If you are not (yet) an expert in Python installations, a main difficulty is choosing the installation procedure.

Some main concepts you need to know before starting:
 * Python itself can be distributed and installed many many ways.
 * Python itself does not contain so many features for scientific computing you need to install "packages".
   numpy, scipy, matplotlib, spikeinterface, ... are Python packages that have a complicated dependency graph between then.
 * packages can be distributed and installed in several ways (pip, conda, uv, mamba, ...)
 * installing many packages at once is challenging (because of their dependency graphs) so you need to do it in an "isolated environement" to not destroy any previous installation. You need to see an "environment" as a sub-installation in a dedicated folder.

Choosing the installer + an environment manager + a package installer is a nightmare for beginners.

The main options are:
  * use "uv", a new, fast and simple package manager. We recommend this for beginners on every operating system.
  * use "anaconda", which does everything. Used to be very popular but theses days it is becoming
    a bad idea because : slow by default and aggressive licensing on the default channel (not always free anymore).
    You need to play with "community channels" to make it free again, which is too complicated for beginners.
    Do not go this way.
  * use Python from the system or Python.org + venv + pip: good and simple idea for linux users.

Here we propose a step by step recipe for beginers based on [**"uv"**](https://github.com/astral-sh/uv).
We used to recommend installing with anaconda. It will be kept here for a while but we do not recommend it anymore.


This environment will install:
 * spikeinterface `full` option
 * spikeinterface-gui
 * kilosort4


### Quick installation using "uv" (recommended)

1. On macOS and Linux. Open a terminal and do
   `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. On Windows. Open a terminal using CMD
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
3. Exit the session and log in again.
4. Download with right click and save this file in your "Documents" folder:
    * [`requirements_stable.txt`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/requirements_stable.txt) for stable release
5. Open terminal or powershell and run:
6. `uv venv si_env --python 3.12`
7. Activate your virtual environment by running:
   - For Mac/Linux: `source si_env/bin/activate` (you should see `(si_env)` in your terminal)
   - For Windows: `si_env\Scripts\activate`
8. Run `uv pip install -r Documents/requirements_stable.txt`


## Installing before release (from source)

Some tools in the spikeinteface ecosystem are getting regular bug fixes (spikeinterface, spikeinterface-gui, probeinterface, neo).
We are making releases 2 to 4 times a year. In between releases if you want to install from source you can use the `requirements_rolling.txt` file to create the environment instead of the `requirements_stable.txt` file. This will install the packages of the ecosystem from source.
This is a good way to test if a patch fixes your issue.


### Check the installation

If you want to test the spikeinterface install you can:

1. Download with right click + save the file [`check_your_install.py`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/check_your_install.py)
    and put it into the "Documents" folder
2. Open the CMD (Windows) or Terminal (Mac/Linux)
3. Activate your si_env : `source si_env/bin/activate` (Max/Linux), `si_env\Scripts\activate` (Windows)
4. Go to your "Documents" folder with `cd Documents` or the place where you downloaded the `check_your_install.py`
5. Run `python check_your_install.py`
6. If you are a Windows user, you should also right click + save [`cleanup_for_windows.py`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/cleanup_for_windows.py). Then transfer `cleanup_for_windows.py` into your "Documents" folder and finally run:
   ```
   python cleanup_for_windows.py
   ```

This script tests the following steps:
  * importing spikeinterface
  * running tridesclous2
  * running kilosort4
  * opening the spikeinterface-gui


### Legacy installation using Anaconda (not recommended anymore)

Steps:

1. Download Anaconda individual edition [here](https://www.anaconda.com/download)
2. Run the installer. Check the box “Add anaconda3 to my Path environment variable”. It makes life easier for beginners.
3. Download with right click + save the file corresponding to your operating system, and put it in "Documents" folder
    * [`full_spikeinterface_environment_windows.yml`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/full_spikeinterface_environment_windows.yml)
    * [`full_spikeinterface_environment_mac.yml`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/full_spikeinterface_environment_mac.yml)
    * [`full_spikeinterface_environment_linux.yml`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/full_spikeinterface_environment_linux.yml)
4. Then open the "Anaconda Command Prompt" (if Windows, search in your applications) or the Terminal (for Mac users)
5. If not in the "Documents" folder type `cd Documents`
6. Then run this depending on your OS:
    * `conda env create --file full_spikeinterface_environment_windows.yml`
    * `conda env create --file full_spikeinterface_environment_mac.yml`


Done! Before running a spikeinterface script you will need to "select" this "environment" with `conda activate si_env`.

Note for **Linux** users: this conda recipe should work but we recommend strongly to use **pip + virtualenv**.


## Installing before release (from source)

Some tools in the spikeinteface ecosystem are getting regular bug fixes (spikeinterface, spikeinterface-gui, probeinterface, neo).
We are making releases 2 to 4 times a year. In between releases if you want to install from source you can use the `full_spikeinterface_environment_rolling_updates.yml` file to create the environment. This will install the packages of the ecosystem from source.
This is a good way to test if a patch fixes your issue.
