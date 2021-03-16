"""
Curation Tutorial
======================

After spike sorting and computing validation metrics, you can automatically curate the spike sorting output using the
quality metrics. This can be done with the :code:`toolkit.curation` submodule.

"""

import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss

#~ ##############################################################################
#~ # First, let's create a toy example:

#~ recording, sorting = se.example_datasets.toy_example(num_channels=4, duration=30, seed=0)

#~ ##############################################################################
#~ # and let's spike sort using klusta

#~ sorting_KL = ss.run_klusta(recording)

#~ print('Units:', sorting_KL.get_unit_ids())
#~ print('Number of units:', len(sorting_KL.get_unit_ids()))

#~ ##############################################################################
#~ # There are several available functions that enables to only retrieve units with respect to some rules. For example,
#~ # let's automatically curate the sorting output so that only the units with SNR > 10 and mean firing rate > 2.3 Hz are
#~ # kept:

#~ sorting_fr = st.curation.threshold_firing_rates(sorting_KL, duration_in_frames=recording.get_num_frames(), threshold=2.3, threshold_sign='less')

#~ print('Units after FR theshold:', sorting_fr.get_unit_ids())
#~ print('Number of units after FR theshold:', len(sorting_fr.get_unit_ids()))

#~ sorting_snr = st.curation.threshold_snrs(sorting_fr, recording, threshold=10, threshold_sign='less')

#~ print('Units after SNR theshold:', sorting_snr.get_unit_ids())
#~ print('Number of units after SNR theshold:', len(sorting_snr.get_unit_ids()))

#~ ##############################################################################
#~ # Let's now check with the :code:`toolkit.validation` submodule that all units have a firing rate > 10 and snr > 0

#~ fr = st.validation.compute_firing_rates(sorting_snr, duration_in_frames=recording.get_num_frames())
#~ snrs = st.validation.compute_snrs(sorting_snr, recording)

#~ print('Firing rates:', fr)
#~ print('SNR:', snrs)
