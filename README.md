[![PyPI version](https://badge.fury.io/py/spikeinterface.svg)](https://badge.fury.io/py/spikeinterface)

# SpikeInterface

SpikeInterface is a Python framework designed to unify preexisting spike sorting technologies into a single code base.

`spikeinterface` is a meta-package that wraps 5 other Python packages from the SpikeInterface team:

- [spikeextractors](https://github.com/SpikeInterface/spikeextractors): Data file I/O and probe handling. [![Build Status](https://travis-ci.org/SpikeInterface/spikeextractors.svg?branch=master)](https://travis-ci.org/SpikeInterface/spikeextractors)
- [spiketoolkit](https://github.com/SpikeInterface/spiketoolkit): Toolkit for pre-processing, post-processing, validation, and automatic curation. [![Build Status](https://travis-ci.org/SpikeInterface/spiketoolkit.svg?branch=master)](https://travis-ci.org/SpikeInterface/spiketoolkit) 
- [spikesorters](https://github.com/SpikeInterface/spikesorters): Python wrappers to spike sorting algorithms. [![Build Status](https://travis-ci.org/SpikeInterface/spikesorters.svg?branch=master)](https://travis-ci.org/SpikeInterface/spikesorters) 
- [spikecomparison](https://github.com/SpikeInterface/spikecomparison): Comparison of spike sorting output (with and without ground-truth). [![Build Status](https://travis-ci.org/SpikeInterface/spikecomparison.svg?branch=master)](https://travis-ci.org/SpikeInterface/spikecomparison) 
- [spikewidgets](https://github.com/SpikeInterface/spikewidgets): Data visualization widgets. [![Build Status](https://travis-ci.org/SpikeInterface/spikewidgets.svg?branch=master)](https://travis-ci.org/SpikeInterface/spikewidgets) 


**On October 8, 2019, we have released the very first beta version of spikeinterface (0.9.1)**

**Please have a look at the [preprint](https://www.biorxiv.org/content/10.1101/796599v1) that describes in detail this project**



## Installation

You can install SpikeInterface from pip:

`pip install spikeinterface` 

Alternatively, you can clone the repository and install from sources the development version:

```bash
git clone https://github.com/SpikeInterface/spikeinterface.git
cd spikeinterface
python setup.py develop
```

**Important**: installing with `python setup.py develop` DOES NOT install the latest version of the different modules.
In order to get the latest updates, clone the above-mentioned repositories and install them from source.

## Documentation

All documentation for SpikeInterface can be found here: https://spikeinterface.readthedocs.io/en/latest/.

You can also check out this 1-hour video tutorial for the NWB User Days 2019:
https://www.youtube.com/watch?v=nWJGwFB7oII


## Citation

If you find SpikeInterface useful in your research, please cite:

```bibtex
@article{buccino2019spikeinterface,
  title={SpikeInterface, a unified framework for spike sorting},
  author={Buccino, Alessio P and Hurwitz, Cole L and Magland, Jeremy and Garcia, Samuel and Siegle, Joshua H and Hurwitz, Roger and Hennig, Matthias H},
  journal={BioRxiv},
  pages={796599},
  year={2019},
  publisher={Cold Spring Harbor Laboratory}
}
```

## SpikeInterface contributors
For any correspondence, contact Alessio Buccino (alessiop.buccino@gmail.com), Cole Hurwitz (colehurwitz@gmail.com), Samuel Garcia (samuel.garcia@cnrs.fr), Jeremy Magland (jmagland@flatironinstitute.org) or Matthias Hennig (m.hennig@ed.ac.uk), or just write an issue!

The following people have contributed code to the project as of 08 June 2020:

* <img src="https://avatars0.githubusercontent.com/u/2369197?v=4" width="16"/> [Achilleas Koutsou](https://github.com/achilleas-k), @G-Node 
* <img src="https://avatars0.githubusercontent.com/u/17097257?v=4" width="16"/> [Alessio Buccino](https://github.com/alejoe91), ETH
* <img src="https://avatars2.githubusercontent.com/u/13655521?v=4" width="16"/> [Alexander Morley](https://github.com/alexmorley), MRC BNDU (University of Oxford)
* <img src="https://avatars0.githubusercontent.com/u/844306?v=4" width="16"/> [Ben Dichter](https://github.com/bendichter), CatalystNeuro
* <img src="https://avatars3.githubusercontent.com/u/31068646?v=4" width="16"/> [Cole Hurwitz](https://github.com/colehurwitz), University of Edinburgh
* <img src="https://avatars1.githubusercontent.com/u/815627?v=4" width="16"/> [Garcia Samuel](https://github.com/samuelgarcia), CNRS, Centre de recherche en neuroscience de Lyon
* <img src="https://avatars2.githubusercontent.com/u/46056408?v=4" width="16"/> [Jasper Wouters](https://github.com/jwouters91)
* <img src="https://avatars2.githubusercontent.com/u/3679296?v=4" width="16"/> [Jeremy Magland](https://github.com/magland)
* <img src="https://avatars3.githubusercontent.com/u/6409964?v=4" width="16"/> [Jose Guzman](https://github.com/JoseGuzman), Austrian Academy of Sciences - OEAW
* <img src="https://avatars3.githubusercontent.com/u/24541631?v=4" width="16"/> [Luiz Tauffer](https://github.com/luiztauffer), @kth
* <img src="https://avatars2.githubusercontent.com/u/11293950?v=4" width="16"/> [Martino Sorbaro](https://github.com/martinosorb), Neuroinformatics, UZH & SynSense.ai
* <img src="https://avatars0.githubusercontent.com/u/5928956?v=4" width="16"/> [Matthias H. Hennig](https://github.com/mhhennig), University of Edinburgh
* <img src="https://avatars2.githubusercontent.com/u/3418096?v=4" width="16"/> [Mikkel Elle Lepperød](https://github.com/lepmik)
* <img src="https://avatars2.githubusercontent.com/u/38734201?v=4" width="16"/> [NMI Biomedical Micro and Nano Engineering](https://github.com/NMI-MSNT)
* <img src="https://avatars1.githubusercontent.com/u/1672447?v=4" width="16"/> [Pierre Yger](https://github.com/yger), Institut de la Vision
* <img src="https://avatars0.githubusercontent.com/u/10051773?v=4" width="16"/> [Roger Hurwitz](https://github.com/rogerhurwitz)
* <img src="https://avatars2.githubusercontent.com/u/15884111?v=4" width="16"/> [Shawn Guo](https://github.com/Shawn-Guo-CN)
* <img src="https://avatars1.githubusercontent.com/u/56535869?v=4" width="16"/> [TRRuikes](https://github.com/TRRuikes)
* <img src="https://avatars3.githubusercontent.com/u/39889?v=4" width="16"/> [Yaroslav Halchenko](https://github.com/yarikoptic), Dartmouth College, @Debian, @DataLad, @PyMVPA, @fail2ban
