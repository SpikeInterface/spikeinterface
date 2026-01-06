from spikeinterface.core import generate_recording
from spikeinterface.preprocessing import detect_artifact_periods


def test_detect_artifact_periods():
    # one segment only
    rec = generate_recording(durations=[10.0, 10])
    artifacts = detect_artifact_periods(rec, method="envelope", 
                                        method_kwargs=dict(detect_threshold=5, freq_max=5.0),
                                        )

if __name__ == "__main__":
    test_detect_artifact_periods()
