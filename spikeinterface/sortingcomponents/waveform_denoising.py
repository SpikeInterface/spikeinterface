import numpy as np
import os
import torch
from torch import nn, optim
import torch.utils.data as Data
from torch.nn import functional as F
from torch import distributions
# from tqdm.auto import tqdm

class SingleChanDenoiser(nn.Module):
    def __init__(self, pretrained_path, n_filters=[16, 8, 4], filter_sizes=[5, 11, 21], spike_size=121):
        super(SingleChanDenoiser, self).__init__()
        feat1, feat2, feat3 = n_filters
        size1, size2, size3 = filter_sizes
        self.conv1 = nn.Sequential(nn.Conv1d(1, feat1, size1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(feat1, feat2, size2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(feat2, feat3, size3), nn.ReLU())
        n_input_feat = feat2 * (spike_size - size1 - size2 + 2)
        self.out = nn.Linear(n_input_feat, spike_size)
        self.pretrained_path = pretrained_path

    def forward(self, x):
        x = x[:, None]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        return self.out(x)

    def load(self):
        checkpoint = torch.load(self.pretrained_path, map_location="cpu")
        self.load_state_dict(checkpoint)
        return self

    

def denoise_wf_nn_single_channel(wf, denoiser, device):
    denoiser = denoiser.to(device)
    n_data, n_times, n_chans = wf.shape
    if wf.shape[0] > 0:
        wf_reshaped = wf.transpose(0, 2, 1).reshape(-1, n_times)
        wf_torch = torch.FloatTensor(wf_reshaped).to(device)
        denoised_wf = denoiser(wf_torch).data
        denoised_wf = denoised_wf.reshape(n_data, n_chans, n_times)
        denoised_wf = denoised_wf.cpu().data.numpy().transpose(0, 2, 1)

        del wf_torch
    else:
        denoised_wf = np.zeros(
            (wf.shape[0], wf.shape[1] * wf.shape[2]), "float32"
        )

    return denoised_wf


def load_nn_and_denoise(wf_array, denoiser_weights_path, architecture_path):
    architecture_denoiser = np.load(architecture_path, allow_pickle = True)
    
    model = SingleChanDenoiser(denoiser_weights_path,
                    architecture_denoiser['n_filters'], 
                    architecture_denoiser['filter_sizes'], 
                    architecture_denoiser['spike_size'])
    denoiser = model.load()
    
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
        
    return denoise_wf_nn_single_channel(wf_array, denoiser, device)
        
