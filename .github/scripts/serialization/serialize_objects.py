#!/usr/bin/env python
"""
Generate cross-version serialization fixtures with the *installed* spikeinterface.

Run this under an OLD spikeinterface release to produce the fixtures that the current
code then loads (see src/spikeinterface/core/tests/test_cross_version_serialization.py):

    python serialize_objects.py [output_dir]

If output_dir is omitted, fixtures are written to ./serialization_fixtures.
The CI workflow cross_version_serialization.yml does this automatically.
"""

import sys
from pathlib import Path

import spikeinterface

sys.path.insert(0, str(Path(__file__).parent))
from objects import OBJECTS, FIXTURE_SUFFIX  # noqa: E402

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("serialization_fixtures")
out_dir.mkdir(parents=True, exist_ok=True)  # do not rmtree an arbitrary path; overwrite in place

print(f"Generating serialization fixtures with spikeinterface {spikeinterface.__version__}")
for entry in OBJECTS:
    obj = entry["build"]()
    for fmt in entry["formats"]:
        dest = out_dir / f"{entry['id']}{FIXTURE_SUFFIX[fmt]}"
        if fmt == "json":
            obj.dump_to_json(dest)
        elif fmt == "pickle":
            obj.dump_to_pickle(dest)
        elif fmt == "binary":
            obj.save(folder=dest, format="binary", overwrite=True)
        elif fmt == "numpy_folder":
            obj.save(folder=dest, format="numpy_folder", overwrite=True)
        elif fmt == "zarr":
            obj.save(folder=dest, format="zarr", overwrite=True)
        print(f"  wrote {dest.name} ({fmt})")

print(f"Fixtures written to: {out_dir.resolve()}")
