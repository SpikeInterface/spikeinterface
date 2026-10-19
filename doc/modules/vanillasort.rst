.. _vanillasort-parameters:

VanillaSort parameters and examples
===================================

VanillaSort accepts a SpikeInterface recording and returns a
:py:class:`~spikeinterface.core.BaseSorting`. This page describes the VanillaSort
0.2.1 API and the parameters exposed by the current SpikeInterface wrapper.
See :ref:`install-vanillasort` for installation and model downloads.

Choose the calling interface
----------------------------

``ss.run_sorter("vanillasort", ...)`` accepts the eight sorter parameters in the
first table below. SpikeInterface validates parameter names against the wrapper's
registered defaults before calling VanillaSort. Extra package parameters such as
``threshold``, ``gate_fn``, and ``target_units`` currently raise ``ValueError``
when passed to this wrapper. Upgrading the VanillaSort package alone does not
extend that registered parameter list.

Use ``vanillasort.sort(recording, ...)`` directly for the additional 0.2.1 options.
Its result is also a SpikeInterface sorting object and can be used with the same
analysis, comparison, and export tools. Custom gate callbacks belong to this
direct Python interface; SpikeInterface saves wrapper parameters as JSON and
does not serialize arbitrary Python functions.

Inspect the wrapper available in your installed SpikeInterface version:

.. code-block:: python

    import spikeinterface.sorters as ss

    defaults = ss.get_default_sorter_params("vanillasort")
    descriptions = ss.get_sorter_params_description("vanillasort")
    for name, default in defaults.items():
        print(name, default, descriptions[name])

The examples below assume ``recording`` contains raw traces sampled at 30,000 Hz
(tolerance 1 Hz), at least 100 samples per segment, and finite 2D channel locations
in micrometres. Each group/shank needs at least four channels. Channel gains and
offsets are applied when available; otherwise traces need a common voltage scale.
Filtering and normalization run inside VanillaSort.

Parameters accepted by run_sorter
---------------------------------

.. list-table:: Registered VanillaSort wrapper parameters
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Meaning
   * - ``model``
     - ``"default"``
     - Published model bundle. Pinned Hugging Face checkpoints are downloaded on
       first inference, verified by SHA-256, and reused from the cache.
   * - ``model_path``
     - ``None``
     - Local combined ``.pt`` checkpoint or directory containing ``config.json``
       and both model checkpoints. Overrides the default model download.
   * - ``device``
     - ``"auto"``
     - ``"auto"`` selects CUDA when available and CPU otherwise. Explicit values
       are ``"cpu"``, ``"cuda"``, and ``"cuda:N"``. MPS is not supported.
   * - ``seed``
     - ``0``
     - Seed for model initialization and GMM clustering. The standalone
       VanillaSort CLI has a separate default of ``30``.
   * - ``components``
     - ``22``
     - Integer GMM component count, at least 2, for each neighborhood and segment.
       Choose it for the recording; 22 is a historical preset. A nonempty
       neighborhood with fewer sortable events than this count raises an error.
   * - ``profile``
     - ``"d1"``
     - ``"d1"`` or ``"canonical"`` preprocessing and threshold profile from the
       loaded model configuration.
   * - ``detector_batch``
     - ``8``
     - Positive number of detector chunks per inference batch. The published
       model uses 2,500-sample chunks. CUDA memory retries reduce this batch size.
   * - ``embedding_batch``
     - ``128``
     - Positive number of waveforms per HuiduRep batch. CUDA memory retries reduce
       this batch size.

.. code-block:: python

    sorting = ss.run_sorter(
        "vanillasort",
        recording,
        folder="vanillasort_output",
        device="auto",
        components=22,
        profile="d1",
        seed=0,
        detector_batch=8,
        embedding_batch=128,
        verbose=True,
    )

``folder`` and ``verbose`` are SpikeInterface runner options. The wrapper manages
VanillaSort's ``output_folder`` and saves package outputs under
``vanillasort_output/sorter_output/inference/``. Do not pass ``output_folder`` as
a sorter parameter. With ``verbose=True``, progress is shown in the terminal and
retained in ``sorter_output/vanillasort.log``; ``spikeinterface_log.json`` is
finalized when sorting finishes.

Additional parameters in vanillasort.sort
-----------------------------------------

The direct API accepts all eight parameters above plus the following options.
The options in this table require VanillaSort 0.2.1 or later and are not currently
registered by the SpikeInterface wrapper.

.. list-table:: Additional direct Python API options
   :header-rows: 1
   :widths: 23 15 62

   * - Parameter
     - Default
     - Meaning and accepted values
   * - ``enable_gate``
     - ``True``
     - Boolean enabling the built-in SNR gate or a supplied custom gate.
       ``False`` keeps score thresholding, peak selection, refractory NMS, and
       main-channel ownership checks while disabling the SNR gate.
   * - ``threshold``
     - ``None``
     - Finite detector sigmoid score in [0, 1], inclusive. Candidates must satisfy
       ``score >= threshold``. ``None`` uses the loaded profile's value:
       published ``d1``: 0.039; ``canonical``: 0.038. This is a score threshold,
       independent of voltage amplitude.
   * - ``direct_keep``
     - ``None``
     - Finite score in [0, 1]. In the built-in gate, eligible candidates with
       ``score >= direct_keep`` bypass the SNR checks. Published defaults are
       0.045 for ``d1`` and 0.041 for ``canonical``. A custom gate replaces this
       bypass together with the SNR rule.
   * - ``primary_snr``
     - ``None``
     - Finite, nonnegative minimum SNR on the strongest channel for the built-in
       gate. ``None`` uses the model setting, published default 3.
   * - ``secondary_snr``
     - ``None``
     - Finite, nonnegative minimum SNR on additional channels for the built-in
       gate. ``None`` uses the model setting, published default 2.
   * - ``min_secondary_channels``
     - ``None``
     - Integer from 1 to 3: how many additional channels must meet
       ``secondary_snr``. ``None`` uses the model setting, published default 1.
   * - ``gate_fn``
     - ``None``
     - Callable ``gate_fn(context)`` returning one boolean per candidate.
       Replaces the built-in SNR/direct-acceptance rule and requires
       ``enable_gate=True``. See the callback example below.
   * - ``neighborhood_mode``
     - ``"dense"``
     - ``"dense"`` retains the original nearest-four neighborhoods.
       ``"cover"`` selects a covering subset with fewer full-record detector
       passes. Every channel remains an input and owns exactly one selected
       patch; group/shank boundaries are respected. Sparser contexts can reduce
       accuracy, so cover mode requires explicit selection.
   * - ``target_units``
     - ``None``
     - Optional positive integer enabling fixed-K merging after cross-patch
       event deduplication. The target counts nonempty output units per segment,
       independently of the local GMM ``components``. K=1 is supported here.
   * - ``postprocessing``
     - ``None``
     - Dictionary of fixed-K policies and thresholds listed below. Requires
       ``target_units``. Set ``target_units`` as its own argument, outside this
       dictionary. Unknown dictionary keys are rejected.

The direct API also accepts ``output_folder=None`` and ``verbose=True``.
``output_folder=None`` keeps the sorting in memory; a supplied path must be new
or empty. ``verbose=False`` suppresses progress. The wrapper sets these two
options through the runner's ``folder`` and ``verbose`` arguments instead.

Thresholds and SNR gating
-------------------------

After peak selection and refractory NMS, the score threshold is applied first.
With the built-in gate enabled, candidates above ``direct_keep`` pass directly.
Remaining candidates need the strongest-channel SNR to meet ``primary_snr`` and
the required number of other channels to meet ``secondary_snr``. These are
per-event negative-peak SNR estimates on filtered, MAD-normalized traces.

Disable the SNR gate while retaining a detector-score threshold:

.. code-block:: python

    import vanillasort

    sorting = vanillasort.sort(
        recording, components=22, threshold=0.139, enable_gate=False,
        output_folder="vanillasort_no_gate", device="auto",
    )

``threshold`` and ``direct_keep`` are independent. If ``threshold >= direct_keep``,
every threshold-passing candidate bypasses the built-in SNR checks. To retain SNR
selection for scores between 0.139 and 0.2, for example, set
``threshold=0.139, direct_keep=0.2, enable_gate=True``. These example values do not
change the defaults. Omitting overrides preserves the loaded model settings.

Custom Python gate
------------------

.. code-block:: python

    import numpy as np
    import vanillasort

    def keep_events(context):
        ranked = np.sort(context.snrs, axis=1)
        return (context.scores >= 0.2) | (
            (ranked[:, -1] >= 3) & (ranked[:, -2] >= 2)
        )

    sorting = vanillasort.sort(
        recording, components=22, threshold=0.139,
        enable_gate=True, gate_fn=keep_events,
    )

``context`` is a ``vanillasort.gating.GateContext`` for one segment/neighborhood.
It contains read-only CPU array copies: ``samples`` and ``scores`` with shape
``(N,)``, ``snrs`` with shape ``(N, 4)``, ``channel_ids`` with shape ``(4,)``, and
``channel_locations`` with shape ``(4, 2)``. Sample indices are relative to the
segment start; SNR columns follow the supplied channel order. Other fields are
``fs_hz``, ``segment_index``, ``neighborhood_index``, ``profile``, and
``event_snr_radius`` in samples.

Return a boolean array or list of shape ``(N,)``. The callback runs once for each
nonempty set of threshold-passing candidates; empty input skips it. Invalid
return types/shapes and callback exceptions stop the run. Callbacks can reject
candidates but cannot restore events removed by thresholding, peak selection,
or NMS. Ownership checks and event deduplication still follow the custom gate.
Only the callable's name is logged; retain its implementation and captured
parameters to reproduce the run.

Fixed-K postprocessing
----------------------

``components`` controls the GMM fitted independently in each neighborhood.
``target_units`` controls subsequent merging into nonempty output units for each
segment, using waveforms on common channels. The target is independent of the
local GMM size. Different segments retain separate unit identities; a target of
10 does not establish 10 shared identities across an entire multisegment record.

.. code-block:: python

    sorting = vanillasort.sort(
        recording,
        components=22,
        target_units=10,
        neighborhood_mode="dense",
        postprocessing={"on_shortfall": "return_available"},
        output_folder="vanillasort_fixed_k",
        device="auto",
    )

The default ``on_shortfall="raise"`` raises
``vanillasort.postprocessing.InsufficientAnchorsError`` when fewer than the target
number of reliable anchors are available. The exception's ``audit`` attribute
describes the shortfall. ``"return_available"`` allows fewer units, including
zero, and records ``exact_k=False``. ``anchor_policy="supplement"`` permits
observed weak or duplicate clusters and marks quality relaxation; it cannot
create missing units. Noise labels and degenerate templates are excluded.

.. list-table:: Keys accepted inside the postprocessing dictionary
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Default
     - Meaning
   * - ``on_shortfall``
     - ``"raise"``
     - ``"raise"`` or ``"return_available"`` controls insufficient-anchor handling.
   * - ``anchor_policy``
     - ``"reliable_only"``
     - ``"reliable_only"`` or ``"supplement"`` controls whether weak/duplicate
       observed clusters may supplement reliable anchors.
   * - ``core_filter``
     - ``True``
     - Boolean enabling reliability criteria: at least 20 events, normalized
       template peak at least 2.5, and absolute peak offset at most 8 samples.
       ``False`` relaxes these criteria and records reduced quality requirements.
   * - ``anchor_only``
     - ``False``
     - Boolean restricting output to selected anchor clusters when true.
       Otherwise compatible fragments may merge into anchors.
   * - ``rank_quality``
     - ``"coherence"``
     - ``"coherence"`` ranks anchors by event count times waveform coherence.
       ``"none"`` uses event count, treating usable clusters as fully coherent.
   * - ``anchor_duplicate_cosine``
     - ``0.995``
     - Finite value at least -1. Anchors meeting this cosine threshold and having
       an amplitude ratio within 1.15 are considered duplicates. Values above 1
       disable this cosine-based duplicate selection criterion.
   * - ``amp_weight``
     - ``0.25``
     - Finite, nonnegative weight of absolute log amplitude ratio in the distance
       used to assign fragments to anchors.
   * - ``min_fragment_coherence``
     - ``0.5``
     - Finite fraction in [0, 1], the minimum waveform coherence for merging a
       non-anchor fragment.
   * - ``min_cosine``
     - ``0.9``
     - Finite value in [-1, 1], the minimum template cosine similarity for merging
       a fragment into its selected anchor.
   * - ``max_amplitude_ratio``
     - ``1.5``
     - Finite value at least 1, bounding the fragment/anchor amplitude ratio in
       either direction.
   * - ``min_margin``
     - ``0.03``
     - Finite, nonnegative minimum difference between the second-best and best
       anchor distances. This criterion does not apply when there is one anchor.

These defaults describe version 0.2.1. Changing them affects anchor selection,
fragment acceptance, and quality; choose settings using the recording's needs.
The independent array utility ``vanillasort.postprocessing.postprocess_fixed_k``
also exists. The recording API prepares its common-channel waveforms automatically.

Outputs and practical constraints
---------------------------------

Spike trains use zero-based segment sample indices, and the ``main_channel_id``
property retains original recording channel IDs. Use
``sorting.get_annotation("detection_gate")`` for resolved gate settings and
``sorting.get_annotation("fixed_k_postprocessing")`` after enabling fixed-K
postprocessing. Saved runs contain ``run.json``, ``events.npz``, and a reloadable
``sorting/`` directory; fixed-K runs also save ``postprocessing_segment*.npz/json``.
On a strict fixed-K shortfall, the JSON diagnostic is written before the error.

Full-segment filtering and normalization require host RAM even with small
inference batches. Runtime grows with duration and neighborhood count because
each neighborhood runs a full detector pass. ``"cover"`` reduces that count while
changing detector contexts. Drift correction and cross-segment identity matching
are not implemented. See the
`VanillaSort 0.2.1 documentation <https://github.com/IgarashiAkatuki/VanillaSort/blob/v0.2.1/README.md>`_
for model configuration, saved diagnostics, and validation scope.
