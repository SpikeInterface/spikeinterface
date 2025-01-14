from spikeinterface.generation import generate_drifting_recording
from spikeinterface.preprocessing.motion import correct_motion
from spikeinterface.sortingcomponents.motion.motion_interpolation import InterpolateMotionRecording

rec = generate_drifting_recording(duration=100)[0]

proc_rec = correct_motion(rec)

rec.set_probe(rec.get_probe())

