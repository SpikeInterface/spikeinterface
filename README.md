# SpikeInterface

SpikeInterface is a Python framework designed to unify preexisting spike sorting technologies into a single code base.

`spikeinterface` is a meta-package that wraps 5 other Python packages from the SpikeInterface team:

- [spikeextractors](https://github.com/SpikeInterface/spikeextractors): data I/O and probe handling
- [spikesorters](https://github.com/SpikeInterface/spikesorters): Python wrappers to spike sorting algorithms
- [spiketoolkit](https://github.com/SpikeInterface/spiketoolkit): toolkit for pre-, post-processing, validation, and automatic curation
- [spikecomparison](https://github.com/SpikeInterface/spikecomparison): comparison of spike sorting output (with and without ground-truth)
- [spikewidgets](https://github.com/SpikeInterface/spikewidgets): visualization widgets


**ALPHA version. The release of an beta version is scheduled for August 31st 2019**


## Installation

Clone the repository and install from sources:

```bash
git clone https://github.com/SpikeInterface/spikeinterface.git
cd spikeinterface
pip install .
```

Soon a `pip install spikeinterface` will be available!

## Documentation

We are currently reorganizing the documentation of each package to a unified documentation.
For now, check out this updated [tutorial](https://github.com/SpikeInterface/spiketutorials/tree/master/Spike_sorting_workshop_2019) for a getting started to SpikeInterface.


### Authors

[Alessio Paolo Buccino](https://www.mn.uio.no/ifi/english/people/aca/alessiob/) - Center for Inegrative Neurolasticity (CINPLA), Department of Biosciences, Physics, and Informatics, University of Oslo, Oslo, Norway

[Cole Hurwitz](https://www.inf.ed.ac.uk/people/students/Cole_Hurwitz.html) - The Institute for Adaptive and Neural Computation (ANC), University of Edinburgh, Edinburgh, Scotland

[Jeremy Magland](https://www.simonsfoundation.org/team/jeremy-magland/) - Center for Computational Biology (CCB), Flatiron Institute, New York, United States

[Matthias Hennig](http://homepages.inf.ed.ac.uk/mhennig/) - The Institute for Adaptive and Neural Computation (ANC), University of Edinburgh, Edinburgh, Scotland

[Samuel Garcia](https://github.com/samuelgarcia) - Centre de Recherche en Neuroscience de Lyon (CRNL), Lyon, France

<br/>
<br/>
For any correspondence, contact Alessio Buccino (alessiop.buccino@gmail.com), Cole Hurwitz (cole.hurwitz@ed.ac.uk), or just write an issue!

