## Installation tips

If you are not (yet) an expert in Python installations, a main difficulty is choosing the installation procedure.
The main ideas you need to know before starting:
 * python itself can be distributed and installed many many ways.
 * python itself does not contain so many features for scientific computing you need to install "packages".
   numpy, scipy, matplotlib, spikeinterface, ... are python packages that have a complicated dependency graph between then.
 * packages  can be distributed and installed in several ways (pip, conda, uv, mamba, ...)
 * installing many packages at once is challenging (because of their dependency graphs) so you need to do it in an "isolated environement"
   to not destroy any previous installation. You need to see an "environment" as a sub installation in a dedicated folder.

Choosing the installer + an environment manager + a package installer is a nightmare for beginners.
The main options are:
  * use "uv" : a new, fast and simple package manager. We recommend this for beginners on every os.
  * use "anaconda", which does everything. Used to be very popular but theses days it is becoming
    a bad idea because : slow by default and aggressive licensing on the default channel (not always free anymore).
    You need to play with "community channels" to make it free again, which is too complicated for beginners.
    Do not go this way.
  * use python from the system or python.org + venv + pip : good and simple idea for linux users.

Here we propose a step by step recipe for beginers based on **"uv"**.
We used to recommend installing with anaconda. It will be kept here for a while but we do not recommend it anymore.


This environment will install:
 * spikeinterface full option
 * spikeinterface-gui
 * kilosort4

Kilosort, Ironclust and HDSort are MATLAB based and need to be installed from source.

### Quick installation using "uv" (recommended)

1. On macOS and Linux. Open a terminal and do
   `curl -LsSf https://astral.sh/uv/install.sh | sh`
1. On windows. Open a terminal using with CMD
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. exit session and log again.
3. Download with right click and save this file corresponding in "Documents" folder:
    * [`requirements_stable.txt`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/requirements_stable.txt)
4. open terminal or powershell
5. `uv venv si_env --python 3.12`
6. For Mac/Linux `source si_env/bin/activate` (you should have `(si_env)` in your terminal)
6. For windows `si_env\Scripts\activate`
7. `uv pip install -r Documents/beginner_requirements_stable.txt` or `uv pip install -r Documents/beginner_requirements_rolling.txt`


More details on [uv here](https://github.com/astral-sh/uv).


## Installing before release

Some tools in the spikeinteface ecosystem are getting regular bug fixes (spikeinterface, spikeinterface-gui, probeinterface, python-neo, sortingview).
We are making releases 2 to 4 times a year. In between releases if you want to install from source you can use the `beginner_requirements_rolling.txt` file to create the environment. This will install the packages of the ecosystem from source.
This is a good way to test if a patch fixes your issue.




### Check the installation


If you want to test the spikeinterface install you can:

1. Download with right click + save the file [`check_your_install.py`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/check_your_install.py)
    and put it into the "Documents" folder
2. Open the CMD (Windows) or Terminal (Mac/Linux)
3. Activate your si_env : `source si_env/bin/activate` (Max/Linux), `si_env\Scripts\activate` (CMD prompt)
4. Go to your "Documents" folder with `cd Documents` or the place where you downloaded the `check_your_install.py`
5. Run this:
    `python check_your_install.py`
6. If a windows user to clean-up you will also need to right click + save [`cleanup_for_windows.py`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/cleanup_for_windows.py)
Then transfer `cleanup_for_windows.py` into your "Documents" folder. Finally run :
   ```
   python cleanup_for_windows.py
   ```

This script tests the following steps:
  * importing spikeinterface
  * running tridesclous2
  * running kilosort4
  * opening the spikeinterface-gui
  * exporting to Phy


### Legacy installation using anaconda (not recommended anymore)

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
