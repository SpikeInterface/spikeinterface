exporter module
===============


export to phy
-------------


folder = 'waveforms_mearec'
we = si.extract_waveforms(recording, sorting, folder,
                          load_if_exists=True,
                          ms_before=1, ms_after=2., max_spikes_per_unit=500,
                          n_jobs=1, chunk_size=30000)
print(we)

output_folder = 'mearec_exported_to_phy'
st.export_to_phy(recording, sorting, output_folder, we,
                 compute_pc_features=False, compute_amplitudes=True,
                 remove_if_exists=True)



export a report
---------------
