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
</table>

SpikeInterface is a Python framework designed to unify preexisting spike sorting technologies into a single code base.



`spikeinterface` version 0.90:

  * breaks backward compatibility with 0.10/0.11/0.12/0.13 series.
  * has been first released in July 2021 (0.90.0)
  * is not a meta-package anymore
  * it doesn't depend on spikeextractors/spiketoolkit/spikesorters/spikecomparison/spikewidgets anymore


**Please have a look at the [eLife paper](https://elifesciences.org/articles/61834) that describes in detail this project**

You can also have a look at the [spikeinterface-gui](https://https://github.com/SpikeInterface/spikeinterface-gui).

## Documentation

All documentation for spikeinterface 0.90 can be found [here](https://spikeinterface.readthedocs.io/en/latest/).

Documentation of current API release 0.12.0 is [here](https://spikeinterface.readthedocs.io/en/stable/).

## How to install new SI version (0.90)

You can install the new `spikeinterface` version with pip:

```bash
pip install spikeinterface==0.90
```

To get the latest updates, you can install `spikeinterface` from sources:

```bash
git clone https://github.com/SpikeInterface/spikeinterface.git
cd spikeinterface
python setup.py install (or develop)
cd ..
```

To install the olde `spikeinterface` API, you can use pip and point to the old version:

```bash
pip install spikeinterface==0.13
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
