# SpikeInterface: a unified framework for spike sorting

<table>
<tr>
  <td>Latest Release</td>
  <td>
    <a href="https://pypi.org/project/spikeinterface/">
    <img src="https://img.shields.io/pypi/v/spikeinterface.svg" alt="latest release" />
    </a>
  </td>
</tr>
<tr>
  <td>Documentation</td>
  <td>
    <a href="https://spikeinterface.readthedocs.io/">
    <img src="https://readthedocs.org/projects/spikeinterface/badge/?version=latest" alt="latest documentation" />
    </a>
  </td>
</tr>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/SpikeInterface/spikeinterface/blob/master/LICENSE">
    <img src="https://img.shields.io/pypi/l/spikeinterface.svg" alt="license" />
    </a>
</td>
</tr>
<tr>
  <td>Build Status</td>
  <td>
    <a href="https://github.com/SpikeInterface/spikeinterface/actions/workflows/full-test-with-codecov.yml/badge.svg">
    <img src="https://github.com/SpikeInterface/spikeinterface/actions/workflows/full-test-with-codecov.yml/badge.svg" alt="CI build status" />
    </a>
  </td>
</tr>
<tr>
	<td>Codecov</td>
	<td>
		<a href="https://codecov.io/github/spikeinterface/spikeinterface">
		<img src="https://codecov.io/gh/spikeinterface/spikeinterface/branch/main/graphs/badge.svg" alt="codecov" />
		</a>
	</td>
</tr>
</table>


[![Twitter](https://img.shields.io/badge/@spikeinterface-%231DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white)](https://twitter.com/spikeinterface) [![Mastodon](https://img.shields.io/badge/-@spikeinterface-%232B90D9?style=for-the-badge&logo=mastodon&logoColor=white)](https://fosstodon.org/@spikeinterface)


SpikeInterface is a Python framework designed to unify preexisting spike sorting technologies into a single code base.

Please [Star](https://github.com/SpikeInterface/spikeinterface/stargazers) the project to support us and [Watch](https://github.com/SpikeInterface/spikeinterface/subscription) to always stay up-to-date!


With SpikeInterface, users can:

- read/write many extracellular file formats.
- pre-process extracellular recordings.
- run many popular, semi-automatic spike sorters (kilosort1-4, mountainsort4-5, spykingcircus,
  tridesclous, ironclust, herdingspikes, yass, waveclus)
- run sorters developed in house (lupin, spkykingcicus2, tridesclous2, simple) that compete with kilosort4
- run theses polar sorters without installation using containers (Docker/Singularity).
- post-process sorted datasets using th SortingAnalyzer
- compare and benchmark spike sorting outputs.
- compute quality metrics to validate and curate spike sorting outputs.
- visualize recordings and spike sorting outputs in several ways (matplotlib, sortingview, jupyter, ephyviewer)
- export a report and/or export to phy
- curate your sorting with several strategies (ml-based, metrics based, manual, ...)
- offer a powerful Qt-based or we-based viewer in a separate package [spikeinterface-gui](https://github.com/SpikeInterface/spikeinterface-gui) for manual curation that replace phy.
- have powerful sorting components to build your own sorter.
- have a full motion/drift correction framework


## Documentation

Detailed documentation of the latest PyPI release of SpikeInterface can be found [here](https://spikeinterface.readthedocs.io/en/stable).

Detailed documentation of the development version of SpikeInterface can be found [here](https://spikeinterface.readthedocs.io/en/latest).

Several tutorials to get started can be found in [spiketutorials](https://github.com/SpikeInterface/spiketutorials).

Checkout our YouTube channel for video tutorials: [SpikeInterface YouTube Channel](https://www.youtube.com/@Spikeinterface).

There are also some useful notebooks [on our blog](https://spikeinterface.github.io) that cover advanced benchmarking
and sorting components.

You can also have a look at the [spikeinterface-gui](https://github.com/SpikeInterface/spikeinterface-gui).


## How to install spikeinterface

You can install the latest version of `spikeinterface` version with pip (using quotes ensures `pip install` works in all terminals/shells):

```bash
pip install "spikeinterface[full]"
```

The `[full]` option installs all the extra dependencies for all the different sub-modules.

To install all interactive widget backends, you can use:

```bash
 pip install "spikeinterface[full,widgets]"
```


To get the latest updates, you can install `spikeinterface` from source:

```bash
git clone https://github.com/SpikeInterface/spikeinterface.git
cd spikeinterface
pip install -e .
cd ..
```


## Citation

If you find SpikeInterface useful in your research, please cite:

```bibtex
@article{buccino2020spikeinterface,
  title={SpikeInterface, a unified framework for spike sorting},
  author={Buccino, Alessio Paolo and Hurwitz, Cole Lincoln and Garcia, Samuel and Magland, Jeremy and Siegle, Joshua H and Hurwitz, Roger and Hennig, Matthias H},
  journal={Elife},
  volume={9},
  pages={e61834},
  year={2020},
  publisher={eLife Sciences Publications Limited}
}
```

Please also cite other relevant papers for the specific components you use.
For a full list of references, please check the [references](https://spikeinterface.readthedocs.io/en/latest/references.html) page.
