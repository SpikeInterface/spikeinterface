from .neural_network_denoiser import SingleChannelDenoiser
from .savgol_denoiser import SavGolDenoiser
from .temporal_pca_denoiser import TemporalPCADenoiser

_methods_list = [SingleChannelDenoiser, SavGolDenoiser, TemporalPCADenoiser]
denoising_methods = {m.name: m for m in _methods_list}
