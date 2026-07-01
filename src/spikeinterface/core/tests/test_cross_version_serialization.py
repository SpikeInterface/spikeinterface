"""
Cross-version serialization test: load objects serialized by an OLD spikeinterface
release with the current code and check they reload correctly.

Fixtures are produced by .github/scripts/serialization/generate_fixtures.py run under
an old release (see .github/workflows/cross_version_serialization.yml). The test skips
when no fixtures are present; set SI_SERIALIZATION_FIXTURES_DIR to a fixtures directory
to run it locally.
"""

import os
import sys
from pathlib import Path

import pytest

from spikeinterface.core import load

# The battery definition is shared with the fixture generator, which runs under old
# installs, so it lives under .github/scripts rather than in the package.
_SCRIPTS_DIR = Path(__file__).parents[4] / ".github" / "scripts" / "serialization"
sys.path.insert(0, str(_SCRIPTS_DIR))
from battery import BATTERY, fixture_relpath  # noqa: E402

FIXTURES_DIR = Path(os.environ.get("SI_SERIALIZATION_FIXTURES_DIR", "serialization_fixtures"))

pytestmark = pytest.mark.skipif(
    not FIXTURES_DIR.exists(),
    reason=(
        f"Serialization fixtures not found at '{FIXTURES_DIR}'. "
        "Generate them with .github/scripts/serialization/generate_fixtures.py under an old "
        "spikeinterface release, or set SI_SERIALIZATION_FIXTURES_DIR to a fixtures directory."
    ),
)

_CASES = [(entry, fmt) for entry in BATTERY for fmt in entry["formats"]]


@pytest.mark.parametrize("entry,fmt", _CASES, ids=[f"{entry['id']}-{fmt}" for entry, fmt in _CASES])
def test_load_old_serialized_object(entry, fmt):
    fixture = FIXTURES_DIR / fixture_relpath(entry["id"], fmt)
    assert fixture.exists(), f"missing fixture {fixture}"
    obj = load(fixture)
    entry["check"](obj)
