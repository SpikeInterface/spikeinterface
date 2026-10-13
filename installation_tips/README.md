## Installation tips

If you are not (yet) an expert in Python installations, the first major hurdle is choosing the installation procedure.

Some key concepts you need to know before starting:
 * Python itself can be distributed and installed many, many ways.
 * Python itself does not contain many features for scientific computing, so you need to install "packages". For example
   numpy, scipy, matplotlib, spikeinterface, ... These are all examples of Python packages that aid in scientific computation.
 * All of these packages have their own dependencies which requires figuring out which versions of the dependencies work for
   the combination of packages you as the user want to use.
 * Packages can be distributed and installed in several ways (pip, conda, uv, mamba, ...) and luckily these methods of installation
   typically take care of solving the dependencies for you!
 * Installing many packages at once is challenging (because of their dependency graphs) so you need to do it in an "isolated environment" to    not destroy any previous installation. You need to see an "environment" as a sub-installation in a dedicated folder.

Choosing the installer + an environment manager + a package installer is a nightmare for beginners.

The main options are:
  * use "uv", a new, fast and simple package manager. We recommend this for beginners on every operating system.
  * use "anaconda" (or its flavors-mamba, miniconda), which does everything. Used to be very popular but theses days it is becoming
    harder to use because it is slow by default and has relatively strict licensing on the default channel (not always free anymore).
    You need to play with "community channels" to make it free again, which is complicated for beginners.
    This way is better for users in organizations that have specific licensing agrees with anaconda already in place.
  * use Python from the system or Python.org + venv + pip: good and simple idea for Linux users, but does require familiarity with
    the Python ecosystem (so good for intermediate users).

Here we propose a step by step recipe for beginners based on [**"uv"**](https://github.com/astral-sh/uv).
We used to recommend installing with anaconda. It will be kept here for a while but we do not recommend it anymore.


This recipe will install:
 * spikeinterface `full` option
 * spikeinterface-gui
 * kilosort4

into our uv venv environment.


### Quick installation using "uv" (recommended)

1. On macOS and Linux. Open a terminal and do
   `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. On Windows. Open an instance of the Powershell (Windows has many options this is the recommended one from uv)
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
3. Exit the session and log in again.
4. Download with right click and save this file in your "Documents" folder:
    * [`beginner_requirements_stable.txt`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/beginner_requirements_stable.txt) for stable release
5. Open terminal or powershell and run:
6. `uv venv si_env --python 3.12`
7. Activate your virtual environment by running:
   - For Mac/Linux: `source si_env/bin/activate` (you should see `(si_env)` in your terminal)
   - For Windows: `si_env\Scripts\activate`
8. Run `uv pip install -r Documents/beginner_requirements_stable.txt`


## Installing before release (from source)

Some tools in the spikeinteface ecosystem are getting regular bug fixes (spikeinterface, spikeinterface-gui, probeinterface, neo).
We are making releases 2 to 4 times a year. In between releases if you want to install from source you can use the `beginner_requirements_rolling.txt` file to create the environment instead of the `beginner_requirements_stable.txt` file. This will install the packages of the ecosystem from source.
This is a good way to test if a patch fixes your issue.


### Check the installation

If you want to test the spikeinterface install you can:

1. Download with right click + save the file [`check_your_install.py`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/check_your_install.py)
    and put it into the "Documents" folder
2. Open the CMD Prompt (Windows)[^1] or Terminal (Mac/Linux)
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
3. Download with right click + save the environment YAML file ([`beginner_conda_env_stable.yml`](https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/installation_tips/beginner_conda_env_stable.yml)) and put it in "Documents" folder
4. Then open the "Anaconda Command Prompt" (if Windows, search in your applications) or the Terminal (for Mac users)
5. If not in the "Documents" folder type `cd Documents`
6. Run this command to create the environment:
   ```bash
   conda env create --file beginner_conda_env_stable.yml
   ```

Done! Before running a spikeinterface script you will need to "select" this "environment" with `conda activate si_env`.

Note for **Linux** users: this conda recipe should work but we recommend strongly to use **pip + virtualenv**.



[^1]: Although uv installation instructions are for the Powershell, our sorter scripts are for the CMD Prompt. After the initial installation with Powershell, any session that will have sorting requires the CMD Prompt. If you do not
plan to spike sort in a session either shell could be used.
