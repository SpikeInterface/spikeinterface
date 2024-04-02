import pytest
import numpy as np
from pathlib import Path

from spikeinterface.core.generate import generate_ground_truth_recording, generate_sorting
from spikeinterface.core import Templates, BaseRecording
from spikeinterface.preprocessing.motion import correct_motion, load_motion_info
from spikeinterface.generation.hybrid_tools import estimate_templates_from_recording, generate_hybrid_recording


def test_generate_hybrid_with_sorting():
    gt_sorting = generate_sorting(durations=[10], num_units=20, sampling_frequency=20000)
    rec, _ = generate_ground_truth_recording(durations=[10], sampling_frequency=20000, sorting=gt_sorting)
    hybrid, _ = generate_hybrid_recording(rec)


def test_generate_hybrid_no_motion():
    rec, sorting = generate_ground_truth_recording(sampling_frequency=20000)
    hybrid, sorting = generate_hybrid_recording(rec)


def test_generate_hybrid_motion(tmp_path):
    rec, sorting = generate_ground_truth_recording(sampling_frequency=20000)
    correct_motion(rec, folder=tmp_path / "motion")
    motion = load_motion_info(tmp_path / "motion")
    hybrid, sorting = generate_hybrid_recording(rec, motion)


def test_estimate_templates(tmp_path):
    rec, sorting = generate_ground_truth_recording(num_units=10, sampling_frequency=20000)
    templates = estimate_templates_from_recording(rec, output_folder=tmp_path / "sc", remove_existing_folder=True)
    assert len(templates.templates_array) > 0


if __name__ == "__main__":
    test_generate_hybrid_no_motion()
    test_generate_hybrid_motion()
    test_estimate_templates()
    test_generate_hybrid_with_sorting()
