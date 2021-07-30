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



`spikeinterface` version 0.90.0:

  * break backward compatility with 0.10/0.11/0.12 series.
  * will be release summer 2021
  * is no more a metapackage
  * no more depend on spikeextractors/spiketoolkit/spikesorters/spikecomparison/spikewidgets


**Please have a look at the [eLife paper](https://elifesciences.org/articles/61834) that describes in detail this project**

## Documentation

All documentation for spikeinterface work-in-progress can be found [here](https://spikeinterface.readthedocs.io/en/latest/).

Documentation of current API release 0.12.0 is [here](https://spikeinterface.readthedocs.io/en/stable/).

## How to install work-in-progress version

Here a simple recipe to install work-in-progress version (0.90.0.dev0):

```
git clone https://github.com/NeuralEnsemble/python-neo.git
cd python-neo
python setup.py install (or develop)
cd ..

git clone https://github.com/SpikeInterface/probeinterface.git
cd probeinterface
python setup.py install (or develop)
cd ..

git clone https://github.com/SpikeInterface/spikeinterface.git
cd spikeinterface
python setup.py install (or develop)
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
