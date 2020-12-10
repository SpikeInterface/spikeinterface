<img src="doc/images/logo.png" class="center" />

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

`spikeinterface` is a meta-package that wraps 5 other Python packages from the SpikeInterface team:

- [spikeextractors](https://github.com/SpikeInterface/spikeextractors): Data file I/O and probe handling. [![Build Status](https://travis-ci.org/SpikeInterface/spikeextractors.svg?branch=master)](https://travis-ci.org/SpikeInterface/spikeextractors)
- [spiketoolkit](https://github.com/SpikeInterface/spiketoolkit): Toolkit for pre-processing, post-processing, validation, and automatic curation. [![Build Status](https://travis-ci.org/SpikeInterface/spiketoolkit.svg?branch=master)](https://travis-ci.org/SpikeInterface/spiketoolkit) 
- [spikesorters](https://github.com/SpikeInterface/spikesorters): Python wrappers to spike sorting algorithms. [![Build Status](https://travis-ci.org/SpikeInterface/spikesorters.svg?branch=master)](https://travis-ci.org/SpikeInterface/spikesorters) 
- [spikecomparison](https://github.com/SpikeInterface/spikecomparison): Comparison of spike sorting output (with and without ground-truth). [![Build Status](https://travis-ci.org/SpikeInterface/spikecomparison.svg?branch=master)](https://travis-ci.org/SpikeInterface/spikecomparison) 
- [spikewidgets](https://github.com/SpikeInterface/spikewidgets): Data visualization widgets. [![Build Status](https://travis-ci.org/SpikeInterface/spikewidgets.svg?branch=master)](https://travis-ci.org/SpikeInterface/spikewidgets) 


**On December 10, 2020, a new version of `spikeinterface` has been released (0.11.0). Check out the [release notes](https://spikeinterface.readthedocs.io/en/latest/whatisnew.html)**

**Please have a look at the [eLife paper](https://elifesciences.org/articles/61834) that describes in detail this project**



## Installation

You can install SpikeInterface from pip:

`pip install spikeinterface` 

The pip installation will install a specific and fixed version of the spikeinterface packages. 

To use the latest updates, install `spikeinterface` and the related packages from source:

```bash
git clone https://github.com/SpikeInterface/spikeinterface.git
cd spikeinterface
python setup.py install (or develop)
```

Then you can install the latest releases of the spikeinterface packages:

```bash
pip install --upgrade spikeextractors spiketoolkit spikesorters spikecomparison spikewidgets
```

You can also install the five packages from sources (e.g. for `spikeextractors`): 

```bash
git clone https://github.com/SpikeInterface/spikeextractors.git
cd spikeextractors
python setup.py install (or develop)
```

## Documentation

All documentation for SpikeInterface can be found [here](https://spikeinterface.readthedocs.io/en/latest/).

You can also check out this [1-hour video tutorial](https://www.youtube.com/watch?v=fvKG_-xQ4D8&t=3364s&ab_channel=NeurodataWithoutBorders) for the NWB User Days 2020:



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

## SpikeInterface contributors
For any correspondence, contact Alessio Buccino (alessiop.buccino@gmail.com), Cole Hurwitz (colehurwitz@gmail.com), Samuel Garcia (samuel.garcia@cnrs.fr), Jeremy Magland (jmagland@flatironinstitute.org) or Matthias Hennig (m.hennig@ed.ac.uk), or just write an issue!

The following people have contributed code to the project as of 10 Nov 2020:

* <img src="https://avatars0.githubusercontent.com/u/2369197?v=4" width="16"/> [Achilleas Koutsou](https://github.com/achilleas-k), @G-Node
* <img src="https://avatars3.githubusercontent.com/u/844464?v=4" width="16"/> [Alan Liddell](https://github.com/aliddell), Vidrio Technologies
* <img src="https://avatars0.githubusercontent.com/u/17097257?v=4" width="16"/> [Alessio Buccino](https://github.com/alejoe91), ETH
* <img src="https://avatars2.githubusercontent.com/u/13655521?v=4" width="16"/> [Alexander Morley](https://github.com/alexmorley), Babylon Health
* <img src="https://avatars0.githubusercontent.com/u/844306?v=4" width="16"/> [Ben Dichter](https://github.com/bendichter), CatalystNeuro
* <img src="https://avatars1.githubusercontent.com/u/51133164?v=4" width="16"/> [Cody Baker](https://github.com/CodyCBakerPhD)
* <img src="https://avatars3.githubusercontent.com/u/31068646?v=4" width="16"/> [Cole Hurwitz](https://github.com/colehurwitz), University of Edinburgh
* <img src="https://avatars1.githubusercontent.com/u/5598671?v=4" width="16"/> [Fernando J. Chaure](https://github.com/ferchaure), University of Buenos Aires
* <img src="https://avatars1.githubusercontent.com/u/815627?v=4" width="16"/> [Garcia Samuel](https://github.com/samuelgarcia), CNRS, Centre de recherche en neuroscience de Lyon
* <img src="https://avatars3.githubusercontent.com/u/8941752?v=4" width="16"/> [James Jun](https://github.com/jamesjun), Facebook, Agios-CTRL
* <img src="https://avatars2.githubusercontent.com/u/46056408?v=4" width="16"/> [Jasper Wouters](https://github.com/jwouters91)
* <img src="https://avatars2.githubusercontent.com/u/3679296?v=4" width="16"/> [Jeremy Magland](https://github.com/magland)
* <img src="https://avatars3.githubusercontent.com/u/6409964?v=4" width="16"/> [Jose Guzman](https://github.com/JoseGuzman), Austrian Academy of Sciences - OEAW
* <img src="https://avatars2.githubusercontent.com/u/200366?v=4" width="16"/> [Josh Siegle](https://github.com/jsiegle), Allen Institute for Brain Science
* <img src="https://avatars3.githubusercontent.com/u/24541631?v=4" width="16"/> [Luiz Tauffer](https://github.com/luiztauffer), @kth
* <img src="https://avatars3.githubusercontent.com/u/7804376?v=4" width="16"/> [Manish Mohapatra](https://github.com/manimoh)
* <img src="https://avatars2.githubusercontent.com/u/11293950?v=4" width="16"/> [Martino Sorbaro](https://github.com/martinosorb), Neuroinformatics, UZH & SynSense.ai
* <img src="https://avatars0.githubusercontent.com/u/5928956?v=4" width="16"/> [Matthias H. Hennig](https://github.com/mhhennig), University of Edinburgh
* <img src="https://avatars2.githubusercontent.com/u/3418096?v=4" width="16"/> [Mikkel Elle Lepperød](https://github.com/lepmik)
* <img src="https://avatars2.githubusercontent.com/u/38734201?v=4" width="16"/> [NMI Biomedical Micro and Nano Engineering](https://github.com/NMI-MSNT)
* <img src="https://avatars1.githubusercontent.com/u/1672447?v=4" width="16"/> [Pierre Yger](https://github.com/yger), Institut de la Vision
* <img src="https://avatars0.githubusercontent.com/u/10051773?v=4" width="16"/> [Roger Hurwitz](https://github.com/rogerhurwitz)
* <img src="https://avatars2.githubusercontent.com/u/11883463?v=4" width="16"/> [Roland Diggelmann](https://github.com/rdiggelmann), ETH Zurich
* <img src="https://avatars2.githubusercontent.com/u/15884111?v=4" width="16"/> [Shawn Guo](https://github.com/Shawn-Guo-CN), School of Informatics, University of Edinburgh
* <img src="https://avatars1.githubusercontent.com/u/56535869?v=4" width="16"/> [TRRuikes](https://github.com/TRRuikes)
* <img src="https://avatars3.githubusercontent.com/u/39889?v=4" width="16"/> [Yaroslav Halchenko](https://github.com/yarikoptic), Dartmouth College, @Debian, @DataLad, @PyMVPA, @fail2ban
* <img src="https://avatars3.githubusercontent.com/u/41306197?v=4" width="16"/> [Michael Scudder](https://github.com/mikeyale)
