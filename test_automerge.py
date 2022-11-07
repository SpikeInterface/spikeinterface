import numpy as np
import probeinterface as pi
import spikeinterface.core as si
import spikeinterface.curation as scur
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre


def load_geometry(recording: si.BaseRecording, prb_file: str) -> si.BaseRecording:
	with open(prb_file, 'r') as file:
		globals = {}
		locals = {}
		exec(file.read(), globals, locals)
		channel_groups = locals['channel_groups']

	probegroup = pi.ProbeGroup()
	for group_id, group in channel_groups.items():
		probe = pi.Probe()

		probe.set_contacts(positions=np.array(group['geometry']))
		probe.set_contact_ids(np.array(group['label'], dtype='S'))
		probe.set_device_channel_indices(group['channels'])
		probegroup.add_probe(probe)

	return recording.set_probegroup(probegroup)


folder = "/mnt/raid0/data/victor_data_injection/data/0170/2020_11_15_eyeblink_1"
recording = si.BinaryRecordingExtractor(f"{folder}/recording/recording_original.dat", 30000, 64, np.int16, gain_to_uV=0.195, offset_to_uV=0)
recording = load_geometry(recording, f"{folder}/../arch.prb")
recording = spre.bandpass_filter(recording, freq_min=150, freq_max=6000, ftype="bessel", filter_order=2)
sorting = se.PhySortingExtractor(f"{folder}/lussac/output/ks2_default")
wvf_extractor = si.extract_waveforms(recording, sorting, folder="/mnt/ssd/tmp/wvfs_automerge", load_if_exists=True,
									 max_spikes_per_unit=2000, chunk_duration='1s', n_jobs=14)

out = scur.get_potential_auto_merge(wvf_extractor)
print(out)
