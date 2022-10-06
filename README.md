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
    <a href="https://travis-ci.org/SpikeInterface/spikeinterface">
    <img src="https://travis-ci.org/SpikeInterface/spikeinterface.svg?branch=master" alt="travis build status" />
    </a>
  </td>
</tr>
<tr>
	<td>Gitter</td>
	<td>
		<a href="https://gitter.im/SpikeInterface/community">
		<img src="https://badges.gitter.im/SpikeInterface.svg" />
	</a>
	</td>
</tr>
<tr>
	<td>Codecov</td>
	<td>
		<a href="https://codecov.io/github/spikeinterface/spikeinterface">
		<img src="https://codecov.io/gh/spikeinterface/spikeinterface/branch/master/graphs/badge.svg" alt="codecov" />
		</a>
	</td>

</tr>
</table>

SpikeInterface is a Python framework designed to unify preexisting spike sorting technologies into a single code base.

With SpikeInterface, users can:

- read/write many extracellular file formats.
- pre-process extracellular recordings.
- run many popular, semi-automatic spike sorters (also in Docker/Singularity containers).
- post-process sorted datasets.
- compare and benchmark spike sorting outputs.
- compute quality metrics to validate and curate spike sorting outputs.
- visualize recordings and spike sorting outputs in several ways (matplotlib, sortingview, in jupyter)
- export report and export to phy
- offer a powerful Qt-based viewer in separate package [spikeinterface-gui](https://https://github.com/SpikeInterface/spikeinterface-gui)
- have some powerful sorting components to build your own sorter.



**Please have a look at the [eLife paper](https://elifesciences.org/articles/61834) that describes in detail this project**

You can also have a look at the [spikeinterface-gui](https://https://github.com/SpikeInterface/spikeinterface-gui).

## Documentation

All documentation for spikeinterface can be found [here](https://spikeinterface.readthedocs.io/en/latest).

Some useful jupyter notebook tutorials can be found in [spiketutorials](https://github.com/SpikeInterface/spiketutorials).

There are also some useful notebooks [on our blog](https://spikeinterface.github.io) that cover advanced benchmarking and sorting components.


## How to install spikeinteface

You can install the new `spikeinterface` version with pip:

```bash
pip install spikeinterface[full]
```

The `[full]` option installs all the extra dependencies for all the different sub-modules. 

To install all interactive widget backends, you can use:

```bash
 pip install spikeinterface[full, widgets]
```


To get the latest updates, you can install `spikeinterface` from sources:

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
