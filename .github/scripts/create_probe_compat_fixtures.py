#!/usr/bin/env python
"""
Creates probe compatibility fixtures using the *currently installed* spikeinterface.

Run this script with spikeinterface==0.104.* installed to produce the fixture
files consumed by test_probe_backward_compat.py:

    python create_probe_compat_fixtures.py [output_dir]

If output_dir is omitted, fixtures are written to ./probe_compat_fixtures.

Note: we use `in_place=True` since a bug (fixed in #4300) prevented probes_info to be properly
saved as annotations in the probegroup when using `in_place=False` in spikeinterface 0.104.*.
"""

import sys
import shutil
import numpy as np
from pathlib import Path

import spikeinterface

print(f"Creating fixtures with spikeinterface {spikeinterface.__version__}")

OUTPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("probe_compat_fixtures")
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)

from probeinterface import generate_linear_probe, ProbeGroup
from spikeinterface.core import NumpyRecording

# -----------------------------------------------------------------------
# Fixture 1: single probe, sequential device_channel_indices
# -----------------------------------------------------------------------
n = 8
probe = generate_linear_probe(num_elec=n, ypitch=20.0)
probe.annotate(name="test_probe", manufacturer="test_vendor")
probe.set_contact_ids([f"e{i}" for i in range(n)])
probe.set_device_channel_indices(np.arange(n))
probe.create_auto_shape()

traces = np.arange(1000 * n, dtype="int16").reshape(1000, n)
rec_single = NumpyRecording([traces], sampling_frequency=30000.0)
rec_single.set_probe(probe, in_place=True)

rec_single_bin = rec_single.save(folder=str(OUTPUT_DIR / "single_probe_binary"))
rec_single_zarr = rec_single.save(folder=str(OUTPUT_DIR / "single_probe.zarr"), format="zarr")
rec_single_bin.dump_to_json(str(OUTPUT_DIR / "single_probe.json"))
rec_single_bin.dump_to_pickle(str(OUTPUT_DIR / "single_probe.pkl"))

# -----------------------------------------------------------------------
# Fixture 2: two probes with per-probe name/manufacturer
# -----------------------------------------------------------------------
n_A, n_B = 8, 8
probe_A = generate_linear_probe(num_elec=n_A, ypitch=20.0)
probe_A.move([0.0, 0.0])
probe_A.annotate(name="probe_A", manufacturer="vendor_X")
probe_A.set_contact_ids([f"a{i}" for i in range(n_A)])
probe_A.set_device_channel_indices(np.arange(n_A))
probe_A.create_auto_shape()

probe_B = generate_linear_probe(num_elec=n_B, ypitch=20.0)
probe_B.move([500.0, 0.0])
probe_B.annotate(name="probe_B", manufacturer="vendor_Y")
probe_B.set_contact_ids([f"b{i}" for i in range(n_B)])
probe_B.set_device_channel_indices(np.arange(n_A, n_A + n_B))
probe_B.create_auto_shape()

pg = ProbeGroup()
pg.add_probe(probe_A)
pg.add_probe(probe_B)

n_total = n_A + n_B
traces2 = np.arange(1000 * n_total, dtype="int16").reshape(1000, n_total)
rec_two = NumpyRecording([traces2], sampling_frequency=30000.0)
rec_two.set_probegroup(pg, in_place=True)

rec_two_bin = rec_two.save(folder=str(OUTPUT_DIR / "two_probe_binary"))
rec_two_zarr = rec_two.save(folder=str(OUTPUT_DIR / "two_probe.zarr"), format="zarr")
rec_two_bin.dump_to_json(str(OUTPUT_DIR / "two_probe.json"))
rec_two_bin.dump_to_pickle(str(OUTPUT_DIR / "two_probe.pkl"))

# -----------------------------------------------------------------------
# Fixture 3: probe with shuffled device_channel_indices
# Verifies that the channel-reordering logic is preserved across versions.
# -----------------------------------------------------------------------
n = 8
probe_sh = generate_linear_probe(num_elec=n, ypitch=20.0)
probe_sh.annotate(name="shuffled_probe", manufacturer="shuffle_vendor")
shuffled_dci = np.array([3, 0, 7, 1, 5, 2, 6, 4])  # permutation of 0..7
probe_sh.set_device_channel_indices(shuffled_dci)

# traces[:, j] corresponds to recording channel j, which after set_probe
# is mapped to the contact whose dci equals j.
traces3 = np.arange(1000 * n, dtype="int16").reshape(1000, n)
rec_sh = NumpyRecording([traces3], sampling_frequency=30000.0)
rec_sh.set_probe(probe_sh, in_place=True)

rec_sh_bin = rec_sh.save(folder=str(OUTPUT_DIR / "shuffled_probe_binary"))
rec_sh_zarr = rec_sh.save(folder=str(OUTPUT_DIR / "shuffled_probe.zarr"), format="zarr")
rec_sh_bin.dump_to_json(str(OUTPUT_DIR / "shuffled_probe.json"))
rec_sh_bin.dump_to_pickle(str(OUTPUT_DIR / "shuffled_probe.pkl"))

print(f"Fixtures written to: {OUTPUT_DIR.resolve()}")

# -----------------------------------------------------------------------
# Fixture 4: two probes with interleaved device_channel_indices
# -----------------------------------------------------------------------
n = 8
probe_A = generate_linear_probe(num_elec=n, ypitch=20.0)
probe_A.move([0.0, 0.0])
probe_A.annotate(name="probe_A", manufacturer="vendor_X")
probe_A.set_contact_ids([f"a{i}" for i in range(n)])
probe_A.set_device_channel_indices(np.arange(0, 2 * n, 2))  # even indices
probe_A.create_auto_shape()

probe_B = generate_linear_probe(num_elec=n, ypitch=20.0)
probe_B.move([500.0, 0.0])
probe_B.annotate(name="probe_B", manufacturer="vendor_Y")
probe_B.set_contact_ids([f"b{i}" for i in range(n)])
probe_B.set_device_channel_indices(np.arange(1, 2 * n, 2))  # odd indices
probe_B.create_auto_shape()

pg = ProbeGroup()
pg.add_probe(probe_A)
pg.add_probe(probe_B)

n_total = 2 * n
traces2 = np.arange(1000 * n_total, dtype="int16").reshape(1000, n_total)
rec_two_inter = NumpyRecording([traces2], sampling_frequency=30000.0)
rec_two_inter.set_probegroup(pg, in_place=True)

rec_two_inter_bin = rec_two_inter.save(folder=str(OUTPUT_DIR / "two_probe_interleaved_binary"))
rec_two_inter_zarr = rec_two_inter.save(folder=str(OUTPUT_DIR / "two_probe_interleaved.zarr"), format="zarr")
rec_two_inter_bin.dump_to_json(str(OUTPUT_DIR / "two_probe_interleaved.json"))
rec_two_inter_bin.dump_to_pickle(str(OUTPUT_DIR / "two_probe_interleaved.pkl"))
