Estimate drift using the LFP traces
===================================

Charlie Windolf and colleagues have developed a method to estimate the
motion using the LFP signal : **dredge**.

You can see more detail in this preprint `DREDge: robust motion
correction for high-density extracellular recordings across
species <https://www.biorxiv.org/content/10.1101/2023.10.24.563768v1>`__

This method is particularly adapated for the open dataset recorded at
Massachusetts General Hospital by Angelique Paulk and colleagues. The
dataset can be dowloaed `on
datadryad <https://datadryad.org/stash/dataset/doi:10.5061/dryad.d2547d840>`__.
This challenging dataset contain recording on patient with neuropixel
probe! But a very high and very fast motion on the probe prevent doing
spike sorting.

The **dredge** method has two sides **dredge_lfp** and **dredge_ap**.
They both haave been ported inside spikeinterface. Here we will use the
**dredge_lfp**.

Here we demonstrate how to use this method to estimate the fast and high
drift on this recording.

For each patient, the dataset contains two recording : a high pass (AP -
30kHz) and a low pass (FP - 2.5kHz). We will use the low pass here.

.. code:: ipython3

    %matplotlib inline
    %load_ext autoreload
    %autoreload 2

.. code:: ipython3

    from pathlib import Path
    import matplotlib.pyplot as plt

    import spikeinterface.full as si
    from spikeinterface.sortingcomponents.motion import estimate_motion

.. code:: ipython3

    # the dataset has been locally downloaded
    base_folder = Path("/mnt/data/sam/DataSpikeSorting/")
    np_data_drift = base_folder / 'human_neuropixel/Pt02/'

read the spikeglx file
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    raw_rec = si.read_spikeglx(np_data_drift)
    print(raw_rec)


.. parsed-literal::

    SpikeGLXRecordingExtractor: 384 channels - 2.5kHz - 1 segments - 2,183,292 samples
                                873.32s (14.56 minutes) - int16 dtype - 1.56 GiB


preprocessing
~~~~~~~~~~~~~

Contrary to **dredge_ap** which need peak and peak location, the
**dredge_lfp** is estimating the motion directly on traces but the
method need an important preprocessing: \* low pass filter : this focus
the signal on a particular band \* phase_shift : this is needed to
conpensate the digitalization unalignement \* resample : the sample
fequency of the signal will be the sample frequency of the estimated
motion. Here we choose 250Hz to have 4ms precission. \*
directional_derivative : this optional step apply a derivative at second
order to enhance edges on the traces. This is not a general rules and
need to be tested case by case. \* average_across_direction : neuropixel
1 probe has several contact per depth. They are average to get a unique
virtual signal along the probe depth (“y” in probeinterface and
spikeinterface).

When appying this preprocessing the motion can be estimated almost by
eyes ont the traces plotted with the map mode.

.. code:: ipython3

    lfprec = si.bandpass_filter(
        raw_rec,
        freq_min=0.5,
        freq_max=250,

        margin_ms=1500.,
        filter_order=3,
        dtype="float32",
        add_reflect_padding=True,
    )
    lfprec = si.phase_shift(lfprec)
    lfprec = si.resample(lfprec, resample_rate=250, margin_ms=1000)

    lfprec = si.directional_derivative(lfprec, order=2, edge_order=1)
    lfprec = si.average_across_direction(lfprec)

    print(lfprec)


.. parsed-literal::

    AverageAcrossDirectionRecording: 192 channels - 0.2kHz - 1 segments - 218,329 samples
                                     873.32s (14.56 minutes) - float32 dtype - 159.91 MiB


.. code:: ipython3

    %matplotlib inline
    si.plot_traces(lfprec, backend="matplotlib", mode="map", clim=(-0.05, 0.05), time_range=(400, 420))




.. parsed-literal::

    <spikeinterface.widgets.traces.TracesWidget at 0x75bc74d0af90>




.. image:: drift_with_lfp_files/drift_with_lfp_8_1.png


Run the method
~~~~~~~~~~~~~~

``estimate_motion()`` is the generic funciton with multi method in
spikeinterface.

This return a ``Motion`` object, you can note that the interval is
exactly the same as downsampled signal.

Here we use ``rigid=True``, this means that we have one unqiue signal to
describe the motion for the entire probe.

.. code:: ipython3

    motion = estimate_motion(lfprec, method='dredge_lfp', rigid=True, progress_bar=True)
    motion



.. parsed-literal::

    Online chunks [10.0s each]:   0%|          | 0/87 [00:00<?, ?it/s]




.. parsed-literal::

    Motion rigid - interval 0.004s - 1 segments



plot the drift
~~~~~~~~~~~~~~

When plotting the drift, we can notice a very fast drift which
corresponf to the heart rate.

This motion match the LFP signal above.

.. code:: ipython3

    fig, ax = plt.subplots()
    si.plot_motion(motion, mode='line', ax=ax)
    ax.set_xlim(400, 420)
    ax.set_ylim(800, 1300)





.. parsed-literal::

    (800.0, 1300.0)




.. image:: drift_with_lfp_files/drift_with_lfp_12_1.png
