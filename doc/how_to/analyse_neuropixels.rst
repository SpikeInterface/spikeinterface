Analyse Neuropixels datasets
============================

This example shows how to perform Neuropixels-specific analysis,
including custom pre- and post-processing.

.. code:: ipython

    %matplotlib inline

.. code:: ipython

    import spikeinterface.full as si

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

.. code:: ipython

    base_folder = Path('/mnt/data/sam/DataSpikeSorting/neuropixel_example/')

    spikeglx_folder = base_folder / 'Rec_1_10_11_2021_g0'


Read the data
-------------

The ``SpikeGLX`` folder can contain several “streams” (AP, LF and NIDQ).
We need to specify which one to read:

.. code:: ipython

    stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
    stream_names




.. parsed-literal::

    ['imec0.ap', 'nidq', 'imec0.lf']



.. code:: ipython

    # we do not load the sync channel, so the probe is automatically loaded
    raw_rec = si.read_spikeglx(spikeglx_folder, stream_name='imec0.ap', load_sync_channel=False)
    raw_rec




.. parsed-literal::

    SpikeGLXRecordingExtractor: 384 channels - 1 segments - 30.0kHz - 1138.145s



.. code:: ipython

    # we automaticaly have the probe loaded!
    raw_rec.get_probe().to_dataframe()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>x</th>
          <th>y</th>
          <th>contact_shapes</th>
          <th>width</th>
          <th>shank_ids</th>
          <th>contact_ids</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>16.0</td>
          <td>0.0</td>
          <td>square</td>
          <td>12.0</td>
          <td></td>
          <td>e0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>48.0</td>
          <td>0.0</td>
          <td>square</td>
          <td>12.0</td>
          <td></td>
          <td>e1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.0</td>
          <td>20.0</td>
          <td>square</td>
          <td>12.0</td>
          <td></td>
          <td>e2</td>
        </tr>
        <tr>
          <th>3</th>
          <td>32.0</td>
          <td>20.0</td>
          <td>square</td>
          <td>12.0</td>
          <td></td>
          <td>e3</td>
        </tr>
        <tr>
          <th>4</th>
          <td>16.0</td>
          <td>40.0</td>
          <td>square</td>
          <td>12.0</td>
          <td></td>
          <td>e4</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>379</th>
          <td>32.0</td>
          <td>3780.0</td>
          <td>square</td>
          <td>12.0</td>
          <td></td>
          <td>e379</td>
        </tr>
        <tr>
          <th>380</th>
          <td>16.0</td>
          <td>3800.0</td>
          <td>square</td>
          <td>12.0</td>
          <td></td>
          <td>e380</td>
        </tr>
        <tr>
          <th>381</th>
          <td>48.0</td>
          <td>3800.0</td>
          <td>square</td>
          <td>12.0</td>
          <td></td>
          <td>e381</td>
        </tr>
        <tr>
          <th>382</th>
          <td>0.0</td>
          <td>3820.0</td>
          <td>square</td>
          <td>12.0</td>
          <td></td>
          <td>e382</td>
        </tr>
        <tr>
          <th>383</th>
          <td>32.0</td>
          <td>3820.0</td>
          <td>square</td>
          <td>12.0</td>
          <td></td>
          <td>e383</td>
        </tr>
      </tbody>
    </table>
    <p>384 rows × 6 columns</p>
    </div>



.. code:: ipython

    fig, ax = plt.subplots(figsize=(15, 10))
    si.plot_probe_map(raw_rec, ax=ax, with_channel_ids=True)
    ax.set_ylim(-100, 100)




.. parsed-literal::

    (-100.0, 100.0)




.. image:: analyse_neuropixels_files/analyse_neuropixels_8_1.png


Preprocess the recording
------------------------

Let’s do something similar to the IBL destriping chain (See
:ref:``ibl_destripe``) to preprocess the data but:

-  instead of interpolating bad channels, we remove then.
-  instead of highpass_spatial_filter() we use common_reference()

.. code:: ipython

    rec1 = si.highpass_filter(raw_rec, freq_min=400.)
    bad_channel_ids, channel_labels = si.detect_bad_channels(rec1)
    rec2 = rec1.remove_channels(bad_channel_ids)
    print('bad_channel_ids', bad_channel_ids)

    rec3 = si.phase_shift(rec2)
    rec4 = si.common_reference(rec3, operator="median", reference="global")
    rec = rec4
    rec


.. parsed-literal::

    bad_channel_ids ['imec0.ap#AP191']




.. parsed-literal::

    CommonReferenceRecording: 383 channels - 1 segments - 30.0kHz - 1138.145s



Visualize the preprocessing steps
---------------------------------

Interactive explore the preprocess steps could de done with this with
the ipywydgets interactive ploter

.. code:: python

   %matplotlib widget
   si.plot_traces({'filter':rec1, 'cmr': rec4}, backend='ipywidgets')

Note that using this ipywydgets make possible to explore diffrents
preprocessing chain wihtout to save the entire file to disk. Everything
is lazy, so you can change the previsous cell (parameters, step order,
…) and visualize it immediatly.

.. code:: ipython

    # here we use static plot using matplotlib backend
    fig, axs = plt.subplots(ncols=3, figsize=(20, 10))

    si.plot_traces(rec1, backend='matplotlib',  clim=(-50, 50), ax=axs[0])
    si.plot_traces(rec4, backend='matplotlib',  clim=(-50, 50), ax=axs[1])
    si.plot_traces(rec, backend='matplotlib',  clim=(-50, 50), ax=axs[2])
    for i, label in enumerate(('filter', 'cmr', 'final')):
        axs[i].set_title(label)



.. image:: analyse_neuropixels_files/analyse_neuropixels_13_0.png


.. code:: ipython

    # plot some channels
    fig, ax = plt.subplots(figsize=(20, 10))
    some_chans = rec.channel_ids[[100, 150, 200, ]]
    si.plot_traces({'filter':rec1, 'cmr': rec4}, backend='matplotlib', mode='line', ax=ax, channel_ids=some_chans)




.. parsed-literal::

    <spikeinterface.widgets.matplotlib.timeseries.TimeseriesPlotter at 0x7fe9275ef0a0>




.. image:: analyse_neuropixels_files/analyse_neuropixels_14_1.png


Should we save the preprocessed data to a binary file?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Depending on the machine, the I/O speed, and the number of times we will
need to “use” the preprocessed recording, we can decide whether it is
convenient to save the preprocessed recording to a file.

Saving is not necessarily a good choice, as it consumes a lot of disk
space and sometimes the writing to disk can be slower than recomputing
the preprocessing chain on-the-fly.

Here, we decide to do save it because Kilosort requires a binary file as
input, so the preprocessed recording will need to be saved at some
point.

Depending on the complexity of the preprocessing chain, this operation
can take a while. However, we can make use of the powerful
parallelization mechanism of SpikeInterface.

.. code:: ipython

    job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True)

    rec = rec.save(folder=base_folder / 'preprocess', format='binary', **job_kwargs)


.. parsed-literal::

    write_binary_recording with n_jobs = 40 and chunk_size = 30000



.. parsed-literal::

    write_binary_recording:   0%|          | 0/1139 [00:00<?, ?it/s]


.. code:: ipython

    # our recording now points to the new binary folder
    rec




.. parsed-literal::

    BinaryFolderRecording: 383 channels - 1 segments - 30.0kHz - 1138.145s



Check spiking activity and drift before spike sorting
-----------------------------------------------------

A good practice before running a spike sorter is to check the “peaks
activity” and the presence of drifts.

SpikeInterface has several tools to:

-  estimate the noise levels
-  detect peaks (prior to sorting)
-  estimate positions of peaks

Check noise level
~~~~~~~~~~~~~~~~~

Noise levels can be estimated on the scaled traces or on the raw
(``int16``) traces.

.. code:: ipython

    # we can estimate the noise on the scaled traces (microV) or on the raw one (which is in our case int16).
    noise_levels_microV = si.get_noise_levels(rec, return_scaled=True)
    noise_levels_int16 = si.get_noise_levels(rec, return_scaled=False)

.. code:: ipython

    fig, ax = plt.subplots()
    _ = ax.hist(noise_levels_microV, bins=np.arange(5, 30, 2.5))
    ax.set_xlabel('noise  [microV]')




.. parsed-literal::

    Text(0.5, 0, 'noise  [microV]')




.. image:: analyse_neuropixels_files/analyse_neuropixels_21_1.png


Detect and localize peaks
~~~~~~~~~~~~~~~~~~~~~~~~~

SpikeInterface includes built-in algorithms to detect peaks and also to
localize their position.

This is part of the **sortingcomponents** module and needs to be
imported explicitly.

The two functions (detect + localize):

-  can be run parallel
-  are very fast when the preprocessed recording is already saved (and a
   bit slower otherwise)
-  implement several methods

Let’s use here the ``locally_exclusive`` method for detection and the
``center_of_mass`` for peak localization:

.. code:: ipython

    from spikeinterface.sortingcomponents.peak_detection import detect_peaks

    job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True)
    peaks = detect_peaks(rec,  method='locally_exclusive', noise_levels=noise_levels_int16,
                         detect_threshold=5, radius_um=50., **job_kwargs)
    peaks



.. parsed-literal::

    detect peaks:   0%|          | 0/1139 [00:00<?, ?it/s]




.. parsed-literal::

    array([(      21, 224, -45., 0), (      36,  84, -34., 0),
           (      40, 103, -30., 0), ..., (34144653,   5, -30., 0),
           (34144662, 128, -30., 0), (34144867, 344, -30., 0)],
          dtype=[('sample_ind', '<i8'), ('channel_ind', '<i8'), ('amplitude', '<f8'), ('segment_ind', '<i8')])



.. code:: ipython3

    from spikeinterface.sortingcomponents.peak_localization import localize_peaks

    peak_locations = localize_peaks(rec, peaks, method='center_of_mass', radius_um=50., **job_kwargs)



.. parsed-literal::

    localize peaks:   0%|          | 0/1139 [00:00<?, ?it/s]


Check for drifts
~~~~~~~~~~~~~~~~

We can *manually* check for drifts with a simple scatter plots of peak
times VS estimated peak depths.

In this example, we do not see any apparent drift.

In case we notice apparent drifts in the recording, one can use the
SpikeInterface modules to estimate and correct motion. See the
documentation for motion estimation and correction for more details.

.. code:: ipython

    # check for drifts
    fs = rec.sampling_frequency
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(peaks['sample_ind'] / fs, peak_locations['y'], color='k', marker='.',  alpha=0.002)




.. parsed-literal::

    <matplotlib.collections.PathCollection at 0x7f7961802a10>




.. image:: analyse_neuropixels_files/analyse_neuropixels_26_1.png


.. code:: ipython

    # we can also use the peak location estimates to have an insight of cluster separation before sorting
    fig, ax = plt.subplots(figsize=(15, 10))
    si.plot_probe_map(rec, ax=ax, with_channel_ids=True)
    ax.set_ylim(-100, 150)

    ax.scatter(peak_locations['x'], peak_locations['y'], color='purple', alpha=0.002)




.. parsed-literal::

    <matplotlib.collections.PathCollection at 0x7f7961701750>




.. image:: analyse_neuropixels_files/analyse_neuropixels_27_1.png


Run a spike sorter
------------------

Even if running spike sorting is probably the most critical part of the
pipeline, in SpikeInterface this is dead-simple: one function.

**Important notes**:

-  most of sorters are wrapped from external tools (kilosort,
   kisolort2.5, spykingcircus, montainsort4 …) that often also need
   other requirements (e.g., MATLAB, CUDA)
-  some sorters are internally developed (spyekingcircus2)
-  external sorter can be run inside a container (docker, singularity)
   WITHOUT pre-installation

Please carwfully read the ``spikeinterface.sorters`` documentation for
more information.

In this example:

-  we will run kilosort2.5
-  we apply no drift correction (because we don’t have drift)
-  we use the docker image because we don’t want to pay for MATLAB :)

.. code:: ipython

    # check default params for kilosort2.5
    si.get_default_sorter_params('kilosort2_5')




.. parsed-literal::

    {'detect_threshold': 6,
     'projection_threshold': [10, 4],
     'preclust_threshold': 8,
     'car': True,
     'minFR': 0.1,
     'minfr_goodchannels': 0.1,
     'nblocks': 5,
     'sig': 20,
     'freq_min': 150,
     'sigmaMask': 30,
     'nPCs': 3,
     'ntbuff': 64,
     'nfilt_factor': 4,
     'NT': None,
     'do_correction': True,
     'wave_length': 61,
     'keep_good_only': False,
     'n_jobs': 40,
     'chunk_duration': '1s',
     'progress_bar': True}



.. code:: ipython

    # run kilosort2.5 without drift correction
    params_kilosort2_5 = {'do_correction': False}

    sorting = si.run_sorter('kilosort2_5', rec, output_folder=base_folder / 'kilosort2.5_output',
                            docker_image=True, verbose=True, **params_kilosort2_5)

.. code:: ipython

    # the results can be read back for futur session
    sorting = si.read_sorter_folder(base_folder / 'kilosort2.5_output')

.. code:: ipython

    # here we have 31 untis in our recording
    sorting




.. parsed-literal::

    KiloSortSortingExtractor: 31 units - 1 segments - 30.0kHz



Post processing
---------------

All the postprocessing step is based on the **WaveformExtractor**
object.

This object combines a ``recording`` and a ``sorting`` object and
extracts some waveform snippets (500 by default) for each units.

Note that we use the ``sparse=True`` option. This option is important
because the waveforms will be extracted only for a few channels around
the main channel of each unit. This saves tons of disk space and speeds
up the waveforms extraction and further processing.

.. code:: ipython

    we = si.extract_waveforms(rec, sorting, folder=base_folder / 'waveforms_kilosort2.5',
                              sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                              **job_kwargs)



.. parsed-literal::

    extract waveforms shared_memory:   0%|          | 0/1139 [00:00<?, ?it/s]



.. parsed-literal::

    extract waveforms memmap:   0%|          | 0/1139 [00:00<?, ?it/s]


.. code:: ipython

    # the WaveformExtractor contains all information and is persistent on disk
    print(we)
    print(we.folder)


.. parsed-literal::

    WaveformExtractor: 383 channels - 31 units - 1 segments
      before:45 after:60 n_per_units:500 - sparse
    /mnt/data/sam/DataSpikeSorting/neuropixel_example/waveforms_kilosort2.5


.. code:: ipython

    # the waveform extractor can be easily loaded back from folder
    we = si.load_waveforms(base_folder / 'waveforms_kilosort2.5')
    we




.. parsed-literal::

    WaveformExtractor: 383 channels - 31 units - 1 segments
      before:45 after:60 n_per_units:500 - sparse



Many additional computations rely on the ``WaveformExtractor``. Some
computations are slower than others, but can be performed in parallel
using the ``**job_kwargs`` mechanism.

Every computation will also be persistent on disk in the same folder,
since they represent waveform extensions.

.. code:: ipython

    _ = si.compute_noise_levels(we)
    _ = si.compute_correlograms(we)
    _ = si.compute_unit_locations(we)
    _ = si.compute_spike_amplitudes(we, **job_kwargs)
    _ = si.compute_template_similarity(we)



.. parsed-literal::

    extract amplitudes:   0%|          | 0/1139 [00:00<?, ?it/s]


Quality metrics
---------------

We have a single function ``compute_quality_metrics(WaveformExtractor)``
that returns a ``pandas.Dataframe`` with the desired metrics.

Please visit the `metrics
documentation <https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html>`__
for more information and a list of all supported metrics.

Some metrics are based on PCA (like
``'isolation_distance', 'l_ratio', 'd_prime'``) and require to estimate
PCA for their computation. This can be achieved with:

``si.compute_principal_components(waveform_extractor)``

.. code:: ipython

    metrics = si.compute_quality_metrics(we, metric_names=['firing_rate', 'presence_ratio', 'snr',
                                                           'isi_violation', 'amplitude_cutoff'])
    metrics


.. parsed-literal::

    /home/samuel.garcia/Documents/SpikeInterface/spikeinterface/spikeinterface/qualitymetrics/misc_metrics.py:511: UserWarning: Units [11, 13, 15, 18, 21, 22] have too few spikes and amplitude_cutoff is set to NaN
      warnings.warn(f"Units {nan_units} have too few spikes and "




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>firing_rate</th>
          <th>presence_ratio</th>
          <th>snr</th>
          <th>isi_violations_ratio</th>
          <th>isi_violations_count</th>
          <th>amplitude_cutoff</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.798668</td>
          <td>1.000000</td>
          <td>1.324698</td>
          <td>4.591437</td>
          <td>10</td>
          <td>0.011528</td>
        </tr>
        <tr>
          <th>1</th>
          <td>9.886261</td>
          <td>1.000000</td>
          <td>1.959527</td>
          <td>5.333803</td>
          <td>1780</td>
          <td>0.000062</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2.849373</td>
          <td>1.000000</td>
          <td>1.467690</td>
          <td>3.859813</td>
          <td>107</td>
          <td>0.002567</td>
        </tr>
        <tr>
          <th>3</th>
          <td>5.404408</td>
          <td>1.000000</td>
          <td>1.253708</td>
          <td>3.519590</td>
          <td>351</td>
          <td>0.000188</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4.772678</td>
          <td>1.000000</td>
          <td>1.722377</td>
          <td>3.947255</td>
          <td>307</td>
          <td>0.001487</td>
        </tr>
        <tr>
          <th>5</th>
          <td>1.802055</td>
          <td>1.000000</td>
          <td>2.358286</td>
          <td>6.403293</td>
          <td>71</td>
          <td>0.001422</td>
        </tr>
        <tr>
          <th>6</th>
          <td>0.531567</td>
          <td>0.888889</td>
          <td>3.359229</td>
          <td>94.320701</td>
          <td>91</td>
          <td>0.004900</td>
        </tr>
        <tr>
          <th>7</th>
          <td>5.400014</td>
          <td>1.000000</td>
          <td>4.653080</td>
          <td>0.612662</td>
          <td>61</td>
          <td>0.000119</td>
        </tr>
        <tr>
          <th>8</th>
          <td>10.563679</td>
          <td>1.000000</td>
          <td>8.267220</td>
          <td>0.073487</td>
          <td>28</td>
          <td>0.000265</td>
        </tr>
        <tr>
          <th>9</th>
          <td>8.181734</td>
          <td>1.000000</td>
          <td>4.546735</td>
          <td>0.730646</td>
          <td>167</td>
          <td>0.000968</td>
        </tr>
        <tr>
          <th>10</th>
          <td>16.839681</td>
          <td>1.000000</td>
          <td>5.094325</td>
          <td>0.298477</td>
          <td>289</td>
          <td>0.000259</td>
        </tr>
        <tr>
          <th>11</th>
          <td>0.007029</td>
          <td>0.388889</td>
          <td>4.032887</td>
          <td>0.000000</td>
          <td>0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>12</th>
          <td>10.184114</td>
          <td>1.000000</td>
          <td>4.780558</td>
          <td>0.720070</td>
          <td>255</td>
          <td>0.000264</td>
        </tr>
        <tr>
          <th>13</th>
          <td>0.005272</td>
          <td>0.222222</td>
          <td>4.627749</td>
          <td>0.000000</td>
          <td>0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>14</th>
          <td>10.047928</td>
          <td>1.000000</td>
          <td>4.984704</td>
          <td>0.771631</td>
          <td>266</td>
          <td>0.000371</td>
        </tr>
        <tr>
          <th>15</th>
          <td>0.107192</td>
          <td>0.888889</td>
          <td>4.248180</td>
          <td>0.000000</td>
          <td>0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>16</th>
          <td>0.535081</td>
          <td>0.944444</td>
          <td>2.326990</td>
          <td>8.183362</td>
          <td>8</td>
          <td>0.000452</td>
        </tr>
        <tr>
          <th>17</th>
          <td>4.650549</td>
          <td>1.000000</td>
          <td>1.998918</td>
          <td>6.391674</td>
          <td>472</td>
          <td>0.000196</td>
        </tr>
        <tr>
          <th>18</th>
          <td>0.077319</td>
          <td>0.722222</td>
          <td>6.619197</td>
          <td>293.942433</td>
          <td>6</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>19</th>
          <td>7.088727</td>
          <td>1.000000</td>
          <td>1.715093</td>
          <td>5.146421</td>
          <td>883</td>
          <td>0.000268</td>
        </tr>
        <tr>
          <th>20</th>
          <td>9.821243</td>
          <td>1.000000</td>
          <td>1.575338</td>
          <td>5.322677</td>
          <td>1753</td>
          <td>0.000059</td>
        </tr>
        <tr>
          <th>21</th>
          <td>0.046567</td>
          <td>0.666667</td>
          <td>5.899877</td>
          <td>405.178035</td>
          <td>3</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>22</th>
          <td>0.094891</td>
          <td>0.722222</td>
          <td>6.476350</td>
          <td>65.051732</td>
          <td>2</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>23</th>
          <td>1.849501</td>
          <td>1.000000</td>
          <td>2.493723</td>
          <td>13.699104</td>
          <td>160</td>
          <td>0.002927</td>
        </tr>
        <tr>
          <th>24</th>
          <td>1.420733</td>
          <td>1.000000</td>
          <td>1.549977</td>
          <td>4.352889</td>
          <td>30</td>
          <td>0.004044</td>
        </tr>
        <tr>
          <th>25</th>
          <td>0.675661</td>
          <td>0.944444</td>
          <td>4.110071</td>
          <td>56.455515</td>
          <td>88</td>
          <td>0.002457</td>
        </tr>
        <tr>
          <th>26</th>
          <td>0.642273</td>
          <td>1.000000</td>
          <td>1.981111</td>
          <td>2.129918</td>
          <td>3</td>
          <td>0.003152</td>
        </tr>
        <tr>
          <th>27</th>
          <td>1.012173</td>
          <td>0.888889</td>
          <td>1.843515</td>
          <td>6.860925</td>
          <td>24</td>
          <td>0.000229</td>
        </tr>
        <tr>
          <th>28</th>
          <td>0.804818</td>
          <td>0.888889</td>
          <td>3.662210</td>
          <td>38.433006</td>
          <td>85</td>
          <td>0.002856</td>
        </tr>
        <tr>
          <th>29</th>
          <td>1.012173</td>
          <td>1.000000</td>
          <td>1.097260</td>
          <td>1.143487</td>
          <td>4</td>
          <td>0.000845</td>
        </tr>
        <tr>
          <th>30</th>
          <td>0.649302</td>
          <td>0.888889</td>
          <td>4.243889</td>
          <td>63.910958</td>
          <td>92</td>
          <td>0.005439</td>
        </tr>
      </tbody>
    </table>
    </div>



Curation using metrics
----------------------

A very common curation approach is to threshold these metrics to select
*good* units:

.. code:: ipython

    amplitude_cutoff_thresh = 0.1
    isi_violations_ratio_thresh = 1
    presence_ratio_thresh = 0.9

    our_query = f"(amplitude_cutoff < {amplitude_cutoff_thresh}) & (isi_violations_ratio < {isi_violations_ratio_thresh}) & (presence_ratio > {presence_ratio_thresh})"
    print(our_query)


.. parsed-literal::

    (amplitude_cutoff < 0.1) & (isi_violations_ratio < 1) & (presence_ratio > 0.9)


.. code:: ipython

    keep_units = metrics.query(our_query)
    keep_unit_ids = keep_units.index.values
    keep_unit_ids




.. parsed-literal::

    array([ 7,  8,  9, 10, 12, 14])



Export final results to disk folder and visulize with sortingview
-----------------------------------------------------------------

In order to export the final results we need to make a copy of the the
waveforms, but only for the selected units (so we can avoid to compute
them again).

.. code:: ipython

    we_clean = we.select_units(keep_unit_ids, new_folder=base_folder / 'waveforms_clean')

.. code:: ipython

    we_clean




.. parsed-literal::

    WaveformExtractor: 383 channels - 6 units - 1 segments
      before:45 after:60 n_per_units:500 - sparse



Then we export figures to a report folder

.. code:: ipython

    # export spike sorting report to a folder
    si.export_report(we_clean, base_folder / 'report', format='png')

.. code:: ipython

    we_clean = si.load_waveforms(base_folder / 'waveforms_clean')
    we_clean




.. parsed-literal::

    WaveformExtractor: 383 channels - 6 units - 1 segments
      before:45 after:60 n_per_units:500 - sparse



And push the results to sortingview webased viewer

.. code:: python

   si.plot_sorting_summary(we_clean, backend='sortingview')
