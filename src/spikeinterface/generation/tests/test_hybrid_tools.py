import pytest
import numpy as np
from pathlib import Path

from spikeinterface.core.generate import generate_ground_truth_recording
from spikeinterface.core import Templates, BaseRecording
from spikeinterface.preprocessing.motion import correct_motion, load_motion_info
from spikeinterface.generation.hybrid_tools import estimate_templates_from_recording, generate_hybrid_recording


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "generation"
else:
    cache_folder = Path("cache_folder") / "generation"


def test_generate_hybrid_no_motion():
    rec, sorting = generate_ground_truth_recording(sampling_frequency=20000)
    hybrid, sorting = generate_hybrid_recording(rec)

def test_generate_hybrid_motion():
    rec, sorting = generate_ground_truth_recording(sampling_frequency=20000)
    correct_motion(rec, folder=cache_folder / "motion")
    motion = load_motion_info(cache_folder / "motion")
    hybrid, sorting = generate_hybrid_recording(rec, motion)

def test_estimate_templates():
    rec, sorting = generate_ground_truth_recording(num_units=10, sampling_frequency=20000)
    templates = estimate_templates_from_recording(rec, output_folder=cache_folder / 'sc', remove_existing_folder=True)
    assert len(templates.templates_array) > 0

if __name__ == "__main__":
    test_generate_hybrid_no_motion()
    test_generate_hybrid_motion()
    test_estimate_templates()