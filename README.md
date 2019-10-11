[![PyPI version](https://badge.fury.io/py/spikeinterface.svg)](https://badge.fury.io/py/spikeinterface)

# SpikeInterface

SpikeInterface is a Python framework designed to unify preexisting spike sorting technologies into a single code base.

`spikeinterface` is a meta-package that wraps 5 other Python packages from the SpikeInterface team:

- [spikeextractors](https://github.com/SpikeInterface/spikeextractors): Data file I/O and probe handling. [![Build Status](https://travis-ci.org/SpikeInterface/spikeextractors.svg?branch=master)](https://travis-ci.org/SpikeInterface/spikeextractors)
- [spiketoolkit](https://github.com/SpikeInterface/spiketoolkit): Toolkit for pre-processing, post-processing, validation, and automatic curation. [![Build Status](https://travis-ci.org/SpikeInterface/spiketoolkit.svg?branch=master)](https://travis-ci.org/SpikeInterface/spiketoolkit) 
- [spikesorters](https://github.com/SpikeInterface/spikesorters): Python wrappers to spike sorting algorithms. [![Build Status](https://travis-ci.org/SpikeInterface/spikesorters.svg?branch=master)](https://travis-ci.org/SpikeInterface/spikesorters) 
- [spikecomparison](https://github.com/SpikeInterface/spikecomparison): Comparison of spike sorting output (with and without ground-truth). [![Build Status](https://travis-ci.org/SpikeInterface/spikecomparison.svg?branch=master)](https://travis-ci.org/SpikeInterface/spikecomparison) 
- [spikewidgets](https://github.com/SpikeInterface/spikewidgets): Data visualization widgets. [![Build Status](https://travis-ci.org/SpikeInterface/spikewidgets.svg?branch=master)](https://travis-ci.org/SpikeInterface/spikewidgets) 


**ALPHA version. The release of an beta version is scheduled for August 31st 2019**


## Installation

You can install SpikeInterface from pip:

`pip install spikeinterface` 

Alternatively, you can clone the repository and install from sources the development version:

```bash
git clone https://github.com/SpikeInterface/spikeinterface.git
cd spikeinterface
python setup.py develop
```

Similarly, you can reinstall all packages listed above.

## Examples

For using SpikeInterface, please checkout these [examples](https://github.com/SpikeInterface/spikeinterface/tree/master/examples).

Also, you can checkout this [tutorial](https://github.com/SpikeInterface/spiketutorials/tree/master/Spike_sorting_workshop_2019) for getting started with SpikeInterface.

## Documentation

All documentation for SpikeInterface can be found here: https://spikeinterface.readthedocs.io/en/latest/.

### Authors

[Alessio Paolo Buccino](https://www.mn.uio.no/ifi/english/people/aca/alessiob/) - Center for Inegrative Neurolasticity (CINPLA), Department of Biosciences, Physics, and Informatics, University of Oslo, Oslo, Norway

[Cole Hurwitz](https://www.inf.ed.ac.uk/people/students/Cole_Hurwitz.html) - The Institute for Adaptive and Neural Computation (ANC), University of Edinburgh, Edinburgh, Scotland

[Jeremy Magland](https://www.simonsfoundation.org/team/jeremy-magland/) - Center for Computational Biology (CCB), Flatiron Institute, New York, United States

[Matthias Hennig](http://homepages.inf.ed.ac.uk/mhennig/) - The Institute for Adaptive and Neural Computation (ANC), University of Edinburgh, Edinburgh, Scotland

[Samuel Garcia](https://github.com/samuelgarcia) - Centre de Recherche en Neuroscience de Lyon (CRNL), Lyon, France

<br/>
For any correspondence, contact Alessio Buccino (alessiop.buccino@gmail.com), Cole Hurwitz (cole.hurwitz@ed.ac.uk), or just write an issue!

