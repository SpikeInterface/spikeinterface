def test_import():
    import spikeinterface.extractors as se
    import spikeinterface.toolkit as st
    import spikeinterface.sorters as ss
    import spikeinterface.comparison as sc
    import spikeinterface.widgets as sw

    # se
    recording, sorting_true = se.example_datasets.toy_example(duration=60, num_channels=4, seed=0)

    # st
    rec_f = st.preprocessing.bandpass_filter(recording)

    # ss
    print(ss.available_sorters())

    # sc
    sc.compare_two_sorters(sorting_true, sorting_true)

    # sw
    sw.plot_timeseries(rec_f)