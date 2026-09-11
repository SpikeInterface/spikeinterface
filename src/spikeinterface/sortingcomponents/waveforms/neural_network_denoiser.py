import json
import warnings
import importlib.util
from pathlib import Path
from typing import List, Optional
import numpy as np

if importlib.util.find_spec("torch") is not None:
    import torch
    from torch import nn

    HAVE_TORCH = True
else:
    HAVE_TORCH = False

if importlib.util.find_spec("huggingface_hub") is not None:
    from huggingface_hub import hf_hub_download, list_repo_files

    HAVE_HUGGINGFACE = True
else:
    HAVE_HUGGINGFACE = False

from spikeinterface.core import BaseRecording
from spikeinterface.core.node_pipeline import PipelineNode, WaveformsNode, find_parent_of_type
from .waveform_utils import to_temporal_representation, from_temporal_representation


class SingleChannelDenoiser(WaveformsNode):
    """
    Denoiser for temporal dimension of waveforms. It takes as input a WaveformsNode and outputs denoised waveforms.

    Parameters
    ----------
    recording: BaseRecording
        The recording object.
    return_output: bool, default True
        Whether to return the output of the node.
    parents: list of PipelineNode, optional
        The parent nodes of this node. Must include a WaveformsNode.
    model_folder: str, optional
        Path to a folder containing the model .pt file and a .json file with temporal parameters
    repo_id: str, optional
        Huggingface repo id to download the model from. Must contain a .pt file and a .json file with temporal parameters
    model_name: str, optional
        Name of the model to use. If there are multiple .pt files in the model_folder, this specifies which one to use.
    """

    def __init__(
        self,
        recording: BaseRecording,
        return_output: bool = True,
        parents: Optional[List[PipelineNode]] = None,
        model_folder: Optional[str] = None,
        repo_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        assert HAVE_TORCH, "To use the SingleChannelDenoiser you need to install torch"
        waveform_node = find_parent_of_type(parents, WaveformsNode)
        if waveform_node is None:
            raise TypeError(f"Model should have a {WaveformsNode.__name__} in its parents")

        super().__init__(
            recording,
            waveform_node.ms_before,
            waveform_node.ms_after,
            return_output=return_output,
            parents=parents,
        )
        if model_folder is None and repo_id is None:
            raise ValueError("You need to specify either model_folder or repo_id")
        if model_folder is not None and repo_id is not None:
            raise ValueError("You cannot specify both model_folder and repo_id")
        spike_size = waveform_node.nbefore + waveform_node.nafter
        # Load model
        self.denoiser, model_relative_path = self.load_model(
            model_folder=model_folder, repo_id=repo_id, model_name=model_name, spike_size=spike_size
        )

        self.assert_model_and_waveform_temporal_match(
            waveform_node, model_folder=model_folder, repo_id=repo_id, model_relative_path=model_relative_path
        )

    def assert_model_and_waveform_temporal_match(
        self,
        waveform_node: WaveformsNode,
        model_relative_path: str,
        model_folder: Optional[str] = None,
        repo_id: Optional[str] = None,
    ):
        """
        Asserts that the model and the waveform extractor have the same temporal parameters
        """
        # Extract temporal parameters from the waveform extractor
        waveforms_ms_before = waveform_node.ms_before
        waveforms_ms_after = waveform_node.ms_after
        waveforms_sampling_frequency = waveform_node.recording.sampling_frequency

        json_file_path = None
        if model_folder is not None:
            json_file_path = Path(model_folder) / str(model_relative_path).replace(".pt", ".json")
        else:
            try:
                filename = str(model_relative_path).replace(".pt", ".json")
                json_file_path = hf_hub_download(repo_id=repo_id, filename=filename)
            except Exception as e:
                warnings.warn(f"Could not download json file from repo {repo_id}. Model might misbehave")

        if json_file_path is None or not Path(json_file_path).exists():
            warnings.warn(f"Could not find json file for model {model_relative_path}. Model might misbehave")
            return

        # Load the json file in the json_file_path_variable
        with open(json_file_path, "r") as json_file:
            model_info = json.load(json_file)

        model_ms_before = model_info.get("ms_before")
        model_ms_after = model_info.get("ms_after")
        model_sampling_frequency = model_info.get("sampling_frequency")
        model_num_samples = model_info.get("num_samples")
        model_nbefore = model_info.get("nbefore")

        if model_num_samples is not None:
            if model_num_samples != waveform_node.nbefore + waveform_node.nafter:
                raise ValueError(
                    f"Model num_samples {model_num_samples} does not match waveform extractor num_samples {waveform_node.num_samples}"
                )
        if model_ms_before is not None:
            if abs(model_ms_before - waveforms_ms_before) > 0.1:
                raise ValueError(
                    f"Difference between model ms_before {model_ms_before} and waveform extractor ms_before {waveforms_ms_before} is too large"
                )
        if model_ms_after is not None:
            if abs(model_ms_after - waveforms_ms_after) > 0.1:
                raise ValueError(
                    f"Difference between model ms_after {model_ms_after} and waveform extractor ms_after {waveforms_ms_after} is too large"
                )
        if model_sampling_frequency is not None:
            if not np.isclose(model_sampling_frequency, waveforms_sampling_frequency, rtol=1e-3):
                raise ValueError(
                    f"Difference between sampling_frequency {model_sampling_frequency} does not match waveform extractor sampling_frequency {waveforms_sampling_frequency}"
                )
        if model_nbefore is not None:
            if abs(model_nbefore - waveform_node.nbefore) > 5:
                raise ValueError(
                    f"Difference between model nbefore {model_nbefore} and waveform extractor nbefore {waveform_node.nbefore} is too large"
                )

    def load_model(
        self,
        model_folder: Optional[str] = None,
        repo_id: Optional[str] = None,
        model_name: Optional[str] = None,
        spike_size: int = 121,
    ):
        if model_folder is not None:
            pt_files = [f for f in Path(model_folder).iterdir("") if f.suffix == ".pt"]
            if len(pt_files) == 1:
                model_path = pt_files[0]
            else:
                if model_name is not None:
                    raise ValueError(f"Multiple models found in {model_folder}. Please specify model_name")
                assert (
                    model_name is not None
                ), "If there are multiple .pt files in the repo, you need to specify the model_name"
                filename = [f for f in pt_files if model_name in f]
                if len(filename) == 0:
                    raise ValueError(f"Model {model_name} not found in repo {repo_id}")
                elif len(filename) > 1:
                    raise ValueError(f"Multiple models found for {model_name} in repo {repo_id}: {filename}")
                else:
                    model_path = filename[0]
            model_relative_path = model_path.relative_to(model_folder)
        else:
            assert HAVE_HUGGINGFACE, "To download models from Huggingface you need to install huggingface_hub"

            repo_filenames = list_repo_files(repo_id=repo_id)

            pt_files = [f for f in repo_filenames if f.endswith(".pt")]
            if len(pt_files) == 1:
                filename = pt_files[0]
            else:
                assert (
                    model_name is not None
                ), "If there are multiple .pt files in the repo, you need to specify the model_name"
                filename = [f for f in pt_files if model_name in f]
                if len(filename) == 0:
                    raise ValueError(f"Model {model_name} not found in repo {repo_id}")
                elif len(filename) > 1:
                    raise ValueError(f"Multiple models found for {model_name} in repo {repo_id}: {filename}")
                else:
                    filename = filename[0]
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            model_relative_path = filename

        denoiser = SingleChannel1dCNNDenoiser(pretrained_path=model_path, spike_size=spike_size)
        denoiser = denoiser.load()
        model_name = Path(model_path).stem
        return denoiser, model_relative_path

    def compute(self, traces, peaks, waveforms):
        num_channels = waveforms.shape[2]

        # Collapse channels and transform to torch tensor
        temporal_waveforms = to_temporal_representation(waveforms)
        temporal_waveforms_tensor = torch.from_numpy(temporal_waveforms).float()

        # Denoise
        denoised_temporal_waveforms = self.denoiser(temporal_waveforms_tensor).detach().numpy()

        # Reconstruct representation with channels
        denoised_waveforms = from_temporal_representation(denoised_temporal_waveforms, num_channels)

        return denoised_waveforms


if HAVE_TORCH:

    class SingleChannel1dCNNDenoiser(nn.Module):
        def __init__(self, pretrained_path=None, n_filters=[16, 8], filter_sizes=[5, 11], spike_size=121):
            super().__init__()

            out_channels_conv1, out_channels_conv_2 = n_filters
            kernel_size_conv1, kernel_size_conv2 = filter_sizes
            self.conv1 = nn.Sequential(nn.Conv1d(1, out_channels_conv1, kernel_size_conv1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv1d(out_channels_conv1, out_channels_conv_2, kernel_size_conv2), nn.ReLU())
            n_input_feat = out_channels_conv_2 * (spike_size - kernel_size_conv1 - kernel_size_conv2 + 2)
            self.out = nn.Linear(n_input_feat, spike_size)
            self.pretrained_path = pretrained_path

        def forward(self, x):
            x = x[:, None]
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.shape[0], -1)
            x = self.out(x)
            return x

        def load(self, device="cpu"):
            checkpoint = torch.load(self.pretrained_path, map_location=device)
            self.load_state_dict(checkpoint)
            return self
