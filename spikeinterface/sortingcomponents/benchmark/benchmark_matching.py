



class BenchmarkMatching:
    def __init__(self, method, params, mearec_file,  **job_kwarg):
        self.mearec_file = mearec_file

        self.recording, self.gt_sorting = si.read_mearec(mearec_file)
        recording_f = si.bandpass_filter(self.recording, dtype='float32')
        recording_f = si.common_reference(recording_f)
        #~ recording = recording.save(folder=rec_folder, n_jobs=20, chunk_size=30000, progress_bar=True)

        self.we = si.extract_waveforms(recording_f, gt_sorting, wf_folder, load_if_exists=True,
                                   ms_before=2.5, ms_after=3.5, max_spikes_per_unit=500,
                                   n_jobs=20, chunk_size=30000, progress_bar=True)
        print(self.we)
   
    def run(self):
        spikes = find_spikes_from_templates(recording, method=method, method_kwargs=method_kwargs, **job_kwargs)
        
        sorting = si.NumpySorting.from_times_labels(spikes['sample_ind'], spikes['cluster_ind'], recording.get_sampling_frequency())
        print(sorting)

        self.comp = si.CollisionGTComparison(gt_sorting, sorting)
    
    def compute_benchmark(self):
        self.metrics = si.compute_quality_metrics(we, metric_names=['snr'], load_if_exists=True)

    def plot(self, title=None):
        
        if title is None:
            title = self.method

        fig, axs = plt.subplots(ncols=3, figsize=(5, 15))
        ax = axs[0]
        ax.set_title(name)
        si.plot_agreement_matrix(self.comp, ax=ax)
        ax.set_title(title)
        
        ax = axs[1]
        si.plot_sorting_performance(self.comp, metrics, performance_name='accuracy', metric_name='snr', ax=axs[1], color='g')
        si.plot_sorting_performance(self.comp, metrics, performance_name='recall', metric_name='snr', ax=axs[1], color='b')
        si.plot_sorting_performance(self.comp, metrics, performance_name='precision', metric_name='snr', ax=axs[1], color='r')
        
        ax.set_ylim(0.8, 1.1)
        ax.legend(['accuracy', 'recall', 'precision'])
        
        ax = axs[2]
        si.plot_comparison_collision_by_similarity(self.comp, templates, ax=ax)

