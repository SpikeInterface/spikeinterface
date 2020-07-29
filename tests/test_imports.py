import matplotlib
matplotlib.rcParams['backend'] = 'agg'
matplotlib.use('agg')


def test_import1():
    import spikeinterface.extractors as se
    import spikeinterface.toolkit as st
    import spikeinterface.sorters as ss
    import spikeinterface.comparison as sc
    import spikeinterface.widgets as sw

    print(se.example_datasets.toy_example)
    print(st.preprocessing.bandpass_filter)
    print(ss.available_sorters)
    print(sc.compare_two_sorters)
    print(sw.plot_timeseries)


def test_import2():
    import spikeinterface as si

    print(si.extractors.example_datasets.toy_example)
    print(si.toolkit.preprocessing.bandpass_filter)
    print(si.sorters.available_sorters)
    print(si.comparison.compare_two_sorters)
    print(si.widgets.plot_timeseries)
    

if __name__ == '__main__':
    test_import1()
    test_import2()
