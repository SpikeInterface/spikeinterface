from .center_of_mass import LocalizeCenterOfMass
from .monopolar import LocalizeMonopolarTriangulation
from .grid import LocalizeGridConvolution

_methods_list = [LocalizeCenterOfMass, LocalizeMonopolarTriangulation, LocalizeGridConvolution]
peak_localization_methods = {m.name: m for m in _methods_list}
