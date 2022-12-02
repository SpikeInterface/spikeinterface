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
    <a href="https://github.com/SpikeInterface/spikeinterface/actions/workflows/full-test.yml/badge.svg">
    <img src="https://github.com/SpikeInterface/spikeinterface/actions/workflows/full-test.yml/badge.svg" alt="travis build status" />
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
<tr>
	<td>Twitter</td>
	<td>
		<a href="https://twitter.com/spikeinterface?ref_src=twsrc%5Etfw" class="twitter-follow-button" data-show-count="false">@spikeinterface
    </a>
	</td>
</tr>
<tr>
	<td>Mastodon</td>
	<td>
		<p>
    <a href="https://fosstodon.org/@spikeinterface" class="mstdn">@spikeinterface@fosstodon.org</span></a>
    </p>
	</td>
</tr>
</table>

SpikeInterface is a Python framework designed to unify preexisting spike sorting technologies into a single code base.

Please <a class="github-button" href="https://github.com/SpikeInterface/spikeinterface" data-color-scheme="no-preference: light_high_contrast; light: light_high_contrast; dark: light_high_contrast;" data-icon="octicon-star" aria-label="Star SpikeInterface/spikeinterface on GitHub">Star</a> the project to support us and <a class="github-button" href="https://github.com/SpikeInterface/spikeinterface/subscription" data-color-scheme="no-preference: light_high_contrast; light: light_high_contrast; dark: light_high_contrast;" data-icon="octicon-eye" aria-label="Watch SpikeInterface/spikeinterface on GitHub">Watch</a> to always stay up-to-date with latest improvements!


With SpikeInterface, users can:

- read/write many extracellular file formats.
- pre-process extracellular recordings.
- run many popular, semi-automatic spike sorters (also in Docker/Singularity containers).
- post-process sorted datasets.
- compare and benchmark spike sorting outputs.
- compute quality metrics to validate and curate spike sorting outputs.
- visualize recordings and spike sorting outputs in several ways (matplotlib, sortingview, in jupyter)
- export report and export to phy
- offer a powerful Qt-based viewer in separate package [spikeinterface-gui](https://github.com/SpikeInterface/spikeinterface-gui)
- have some powerful sorting components to build your own sorter.


## Documentation

Detailed documentation for spikeinterface can be found [here](https://spikeinterface.readthedocs.io/en/latest).

Several tutorials to get started can be found in [spiketutorials](https://github.com/SpikeInterface/spiketutorials).

There are also some useful notebooks [on our blog](https://spikeinterface.github.io) that cover advanced benchmarking 
and sorting components.

You can also have a look at the [spikeinterface-gui](https://github.com/SpikeInterface/spikeinterface-gui).


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
