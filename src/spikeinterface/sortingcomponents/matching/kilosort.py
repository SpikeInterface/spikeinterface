"""Sorting components: template matching."""

from __future__ import annotations


import numpy as np

from .base import BaseTemplateMatching, _base_matching_dtype

try:
    import torch

    HAVE_TORCH = True
    from torch.nn.functional import conv1d, max_pool2d, max_pool1d
except ImportError:
    HAVE_TORCH = False


spike_dtype = [
    ("sample_index", "int64"),
    ("channel_index", "int64"),
    ("cluster_index", "int64"),
    ("amplitude", "float64"),
    ("segment_index", "int64"),
]


class KiloSortPeeler(BaseTemplateMatching):

    def __init__(
        self,
        recording,
        return_output=True,
        parents=None,
        templates=None,
        temporal_components=None,
        spatial_components=None,
        Th=8,
        max_iter=100,
        engine="torch",
        torch_device="cpu",
    ):

        import scipy

        BaseTemplateMatching.__init__(self, recording, templates, return_output=True, parents=None)
        self.templates_array = self.templates.get_dense_templates()
        self.spatial_components = spatial_components
        self.temporal_components = temporal_components
        self.Th = Th
        self.max_iter = max_iter
        self.engine = "torch"
        self.torch_device = torch_device

        self.num_components = len(self.temporal_components)
        self.num_templates = len(self.templates_array)
        self.num_channels = recording.get_num_channels()
        self.num_samples = self.templates_array.shape[1]

        U = np.zeros((self.num_templates, self.num_channels, self.num_components), dtype=np.float32)
        for i in range(self.num_templates):
            U[i] = np.dot(spatial_components, self.templates_array[i]).T

        Uex = np.einsum("xyz, zt -> xty", U, self.spatial_components)
        if self.engine == "torch":
            Uex = torch.as_tensor(Uex, device=self.torch_device)
            temporal_components_torch = torch.as_tensor(temporal_components, device=self.torch_device)
            X = Uex.reshape(-1, self.num_channels).T
            X = conv1d(X.unsqueeze(1), temporal_components_torch.unsqueeze(1), padding=self.num_samples // 2)
            X = X[:, :, : self.num_templates * self.num_samples]
            Xmax = X.abs().max(0)[0].max(0)[0].reshape(-1, self.num_samples)
            imax = torch.argmax(Xmax, 1)
            Unew_torch = Uex.clone()
            for j in range(self.num_samples):
                ix = imax == j
                Unew_torch[ix] = torch.roll(Unew_torch[ix], self.num_samples // 2 - j, -2)
            self.U = torch.einsum(
                "xty, zt -> xzy", Unew_torch, torch.as_tensor(spatial_components, device=self.torch_device)
            )
            self.W = torch.as_tensor(self.spatial_components, device=self.torch_device)
            WtW = conv1d(
                self.W.reshape(-1, 1, self.num_samples),
                self.W.reshape(-1, 1, self.num_samples),
                padding=self.num_samples,
            )
            WtW = torch.flip(
                WtW,
                [
                    2,
                ],
            )
            UtU = torch.einsum("ikl, jml -> ijkm", self.U, self.U)
            self.ctc = torch.einsum("ijkm, kml -> ijl", UtU, WtW)
            self.trange = torch.arange(-self.num_samples, self.num_samples + 1, device=self.torch_device)
        else:
            X = Uex.reshape(-1, self.num_channels).T
            X = scipy.signal.oaconvolve(X[:, None, :], self.temporal_components[None, :, ::-1], mode="full", axes=2)
            X = X[:, :, self.num_samples // 2 : self.num_samples // 2 + self.num_samples * self.num_templates]
            Xmax = np.abs(X).max(0).max(0).reshape(-1, self.num_samples)
            imax = np.argmax(Xmax, 1)
            Unew = Uex.copy()
            for j in range(self.num_samples):
                ix = imax == j
                Unew[ix] = np.roll(Unew[ix], self.num_samples // 2 - j, -2)

            self.U = np.einsum("xty, zt -> xzy", Unew, spatial_components)
            self.W = self.spatial_components
            WtW = scipy.signal.oaconvolve(self.W[None, :, ::-1], self.W[:, None, :], mode="full", axes=2)
            WtW = np.flip(WtW, 2)
            UtU = np.einsum("ikl, jml -> ijkm", self.U, self.U)
            self.ctc = np.einsum("ijkm, kml -> ijl", UtU, WtW)
            self.trange = np.arange(-self.num_samples, self.num_samples + 1, device=self.torch_device)

        self.nbefore = self.templates.nbefore
        self.nafter = self.templates.nafter
        self.margin = self.num_samples
        self.nm = (self.U**2).sum(-1).sum(-1)

    def get_trace_margin(self):
        return self.margin

    def compute_matching(self, traces, start_frame, end_frame, segment_index):

        if self.engine == "torch":
            X = torch.as_tensor(traces.T, device=self.torch_device)
            B = conv1d(X.unsqueeze(1), self.W.unsqueeze(1), padding=self.num_samples // 2)
            B = torch.einsum("ijk, kjl -> il", self.U, B)

            spikes = np.empty(traces.size, dtype=spike_dtype)
            k = 0

            for t in range(self.max_iter):
                Cf = torch.relu(B) ** 2 / self.nm.unsqueeze(-1)
                Cf[:, : self.num_samples] = 0
                Cf[:, -self.num_samples :] = 0

                Cfmax, imax = torch.max(Cf, 0)
                Cmax = max_pool1d(
                    Cfmax.unsqueeze(0).unsqueeze(0), (2 * self.num_samples + 1), stride=1, padding=(self.num_samples)
                )
                cnd1 = Cmax[0, 0] > self.Th**2
                cnd2 = torch.abs(Cmax[0, 0] - Cfmax) < 1e-9
                xs = torch.nonzero(cnd1 * cnd2)

                if len(xs) == 0:
                    break

                iX = xs[:, :1]
                iY = imax[iX]

                nsp = len(iX)
                spikes[k : k + nsp]["sample_index"] = iX[:, 0].cpu()
                spikes[k : k + nsp]["cluster_index"] = iY[:, 0].cpu()
                amp = B[iY, iX] / self.nm[iY]

                n = 2
                for j in range(n):
                    B[:, iX[j::n] + self.trange] -= amp[j::n] * self.ctc[:, iY[j::n, 0], :]

                spikes[k : k + nsp]["amplitude"] = amp[:, 0].cpu()
                k += nsp

            spikes = spikes[:k]
            spikes["channel_index"] = 0
            spikes["sample_index"] += self.nbefore
            order = np.argsort(spikes["sample_index"])
            spikes = spikes[order]

        else:
            B = scipy.signal.oaconvolve(traces.T[np.newaxis, :, :], self.W[:, None, ::-1], mode="full", axes=2)
            B = np.einsum("ijk, kjl -> il", self.U, B)

        return spikes
