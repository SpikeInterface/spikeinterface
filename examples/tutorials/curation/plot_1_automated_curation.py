"""
Model-based curation tutorial
=============================

Sorters are not perfect. They output excellent units, as well as noisy ones, and ones that
should be split or merged. Hence one should curate the generated units. Historically, this
has been done using laborious manual curation. An alternative is to use automated methods
based on metrics which quantify features of the units. In spikeinterface these are the
quality metrics and the template metrics. A simple approach is to use thresholding:
only accept units whose metrics pass a certain quality threshold. Another approach is to
take one (or more) manually labelled sortings, whose metrics have been computed, and train
a machine learning model to predict labels.

This notebook provides a step-by-step guide on how to take a machine learning model that
someone else has trained and use it to curate your own spike sorted output. SpikeInterface
also provides the tools to train your own model,
`which you can learn about here <https://spikeinterface.readthedocs.io/en/latest/tutorials/curation/plot_2_train_a_model.html>`_.

We'll download a toy model and use it to label our sorted data. We start by importing some packages
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spikeinterface.core as si
import spikeinterface.curation as sc
import spikeinterface.widgets as sw

# note: you can use more cores using e.g.
# si.set_global_jobs_kwargs(n_jobs = 8)

##############################################################################
# Download a pretrained model
# ---------------------------
#
# Let's download a pretrained model from `Hugging Face <https://huggingface.co/>`_ (HF),
# a model sharing platform focused on AI and ML models and datasets. The
# ``load_model`` function allows us to download directly from HF, or use a model in a local
# folder. The function downloads the model and saves it in a temporary folder and returns a
# model and some metadata about the model.

model, model_info = sc.load_model(
    repo_id = "SpikeInterface/toy_tetrode_model",
    trusted = ['numpy.dtype']
)


##############################################################################
# This model was trained on artifically generated tetrode data. There are also models trained
# on real data, like the one discussed `below <#A-model-trained-on-real-Neuropixels-data>`_.
# Each model object has a nice html representation, which will appear if you're using a Jupyter notebook.

model

##############################################################################
# This tells us more information about the model. The one we've just downloaded was trained used
# a ``RandomForestClassifier```. You can also discover this information by running
# ``model.get_params()``. The model object (an `sklearn Pipeline <https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html>`_) also contains information
# about which metrics were used to compute the model. We can access it from the model (or from the model_info)

print(model.feature_names_in_)

##############################################################################
# Hence, to use this model we need to create a ``sorting_analyzer`` with all these metrics computed.
# We'll do this by generating a recording and sorting, creating a sorting analyzer and computing a
# bunch of extensions. Follow these links for more info on `recordings <https://spikeinterface.readthedocs.io/en/latest/modules/extractors.html>`_, `sortings <https://spikeinterface.readthedocs.io/en/latest/modules/sorters.html>`_, `sorting analyzers <https://spikeinterface.readthedocs.io/en/latest/tutorials/core/plot_4_sorting_analyzer.html#sphx-glr-tutorials-core-plot-4-sorting-analyzer-py>`_
# and `extensions <https://spikeinterface.readthedocs.io/en/latest/modules/postprocessing.html>`_.

recording, sorting = si.generate_ground_truth_recording(num_channels=4, seed=4, num_units=10)
sorting_analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording)
sorting_analyzer.compute(['noise_levels','random_spikes','waveforms','templates','spike_locations','spike_amplitudes','correlograms','principal_components','quality_metrics','template_metrics'])
sorting_analyzer.compute('template_metrics', include_multi_channel_metrics=True)

##############################################################################
# This sorting_analyzer now contains the required quality metrics and template metrics.
# We can check that this is true by accessing the extension data.

all_metric_names = list(sorting_analyzer.get_extension('quality_metrics').get_data().keys()) + list(sorting_analyzer.get_extension('template_metrics').get_data().keys())
print(set(model.feature_names_in_).issubset(set(all_metric_names)))

##############################################################################
# Great! We can now use the model to predict labels. Here, we pass the HF repo id directly
# to the ``model_based_label_units`` function. This returns a dictionary containing a label and
# a confidence for each unit contained in the ``sorting_analyzer``.

labels = sc.model_based_label_units(
    sorting_analyzer = sorting_analyzer,
    repo_id = "SpikeInterface/toy_tetrode_model",
    trusted = ['numpy.dtype']
)

print(labels)


##############################################################################
# The model has labelled one unit as bad. Let's look at that one, and also the 'good' unit
# with the highest confidence of being 'good'.

sw.plot_unit_templates(sorting_analyzer, unit_ids=['7','9'])

##############################################################################
# Nice! Unit 9 looks more like an expected action potential waveform while unit 7 doesn't,
# and it seems reasonable that unit 7 is labelled as `bad`. However, for certain experiments
# or brain areas, unit 7 might be a great small-amplitude unit. This example highlights that
# you should be careful applying models trained on one dataset to your own dataset. You can
# explore the currently available models on the `spikeinterface hugging face hub <https://huggingface.co/SpikeInterface>`_
# page, or `train your own one <https://spikeinterface.readthedocs.io/en/latest/tutorials/curation/plot_2_train_a_model.html>`_.
#
# Assess the model performance
# ----------------------------
#
# To assess the performance of the model relative to labels assigned by a human creator, we can load or generate some
# "human labels", and plot a confusion matrix of predicted vs human labels for all clusters. Here
# we'll be a conservative human, who has labelled several units with small amplitudes as 'bad'.

human_labels = ['bad', 'good', 'good', 'bad', 'good', 'bad', 'good', 'bad', 'good', 'good']

# Note: if you labelled using phy, you can load the labels using:
# human_labels = sorting_analyzer.sorting.get_property('quality')
# We need to load in the `label_conversion` dictionary, which converts integers such
# as '0' and '1' to readable labels such as 'good' and 'bad'. This is stored as
# in `model_info`, which we loaded earlier.

from sklearn.metrics import confusion_matrix, balanced_accuracy_score

label_conversion = model_info['label_conversion']
predictions = labels['prediction']

conf_matrix = confusion_matrix(human_labels, predictions)

# Calculate balanced accuracy for the confusion matrix
balanced_accuracy = balanced_accuracy_score(human_labels, predictions)

plt.imshow(conf_matrix)
for (index, value) in np.ndenumerate(conf_matrix):
    plt.annotate( str(value), xy=index, color="white", fontsize="15")
plt.xlabel('Predicted Label')
plt.ylabel('Human Label')
plt.xticks(ticks = [0, 1], labels = list(label_conversion.values()))
plt.yticks(ticks = [0, 1], labels = list(label_conversion.values()))
plt.title('Predicted vs Human Label')
plt.suptitle(f"Balanced Accuracy: {balanced_accuracy}")
plt.show()


##############################################################################
# Here, there are several false positives (if we consider the human labels to be "the truth").
#
# Next, we can also see how the model's confidence relates to the probability that the model
# label matches the human label.
#
# This could be used to help decide which units should be auto-curated and which need further
# manual creation. For example, we might accept any unit as 'good' that the model predicts
# as 'good' with confidence over a threshold, say 80%. If the confidence is lower we might decide to take a
# look at this unit manually. Below, we will create a plot that shows how the agreement
# between human and model labels changes as we increase the confidence threshold. We see that
# the agreement increases as the confidence does. So the model gets more accurate with a
# higher confidence threshold, as expceted.


def calculate_moving_avg(label_df, confidence_label, window_size):

    label_df[f'{confidence_label}_decile'] = pd.cut(label_df[confidence_label], 10, labels=False, duplicates='drop')
    # Group by decile and calculate the proportion of correct labels (agreement)
    p_label_grouped = label_df.groupby(f'{confidence_label}_decile')['model_x_human_agreement'].mean()
    # Convert decile to range 0-1
    p_label_grouped.index = p_label_grouped.index / 10
    # Sort the DataFrame by confidence scores
    label_df_sorted = label_df.sort_values(by=confidence_label)

    p_label_moving_avg = label_df_sorted['model_x_human_agreement'].rolling(window=window_size).mean()

    return label_df_sorted[confidence_label], p_label_moving_avg

confidences = labels['probability']

# Make dataframe of human label, model label, and confidence
label_df = pd.DataFrame(data = {
    'human_label': human_labels,
    'decoder_label': predictions,
    'confidence': confidences},
    index = sorting_analyzer.sorting.get_unit_ids())

# Calculate the proportion of agreed labels by confidence decile
label_df['model_x_human_agreement'] = label_df['human_label'] == label_df['decoder_label']

p_agreement_sorted, p_agreement_moving_avg = calculate_moving_avg(label_df, 'confidence', 3)

# Plot the moving average of agreement
plt.figure(figsize=(6, 6))
plt.plot(p_agreement_sorted, p_agreement_moving_avg, label = 'Moving Average')
plt.axhline(y=1/len(np.unique(predictions)), color='black', linestyle='--', label='Chance')
plt.xlabel('Confidence'); #plt.xlim(0.5, 1)
plt.ylabel('Proportion Agreement with Human Label'); plt.ylim(0, 1)
plt.title('Agreement vs Confidence (Moving Average)')
plt.legend(); plt.grid(True); plt.show()

##############################################################################
# In this case, you might decide to only trust labels which had confidence over above 0.88,
# and manually labels the ones the model isn't so confident about.
#
# A model trained on real Neuropixels data
# ----------------------------------------
#
# Above, we used a toy model trained on generated data. There are also models on HuggingFace
# trained on real data.
#
# For example, the following classifiers are trained on Neuropixels data from 11 mice recorded in
# V1,SC and ALM: https://huggingface.co/SpikeInterface/UnitRefine_noise_neural_classifier/ and
# https://huggingface.co/SpikeInterface/UnitRefine_sua_mua_classifier/. One will classify units into
# `noise` or `neural` and the other will classify the `neural` units into single
# unit activity (sua) units and multi-unit activity (mua) units.
#
# There is more information about the model on the model's HuggingFace page. Take a look!
# The idea here is to first apply the noise/neural classifier, then the sua/mua one.
# We can do so as follows:
#

# Apply the noise/neural model
noise_neuron_labels = sc.model_based_label_units(
    sorting_analyzer=sorting_analyzer,
    repo_id="SpikeInterface/UnitRefine_noise_neural_classifier",
    trust_model=True,
)

noise_units = noise_neuron_labels[noise_neuron_labels['prediction']=='noise']
analyzer_neural = sorting_analyzer.remove_units(noise_units.index)

# Apply the sua/mua model
sua_mua_labels = sc.model_based_label_units(
    sorting_analyzer=analyzer_neural,
    repo_id="SpikeInterface/UnitRefine_sua_mua_classifier",
    trust_model=True,
)

all_labels = pd.concat([sua_mua_labels, noise_units]).sort_index()
print(all_labels)

##############################################################################
# Both steps can be done in one go using the ``unitrefine_label_units`` function:
#

all_labels = sc.unitrefine_label_units(
    sorting_analyzer,
    noise_neural_classifier="SpikeInterface/UnitRefine_noise_neural_classifier",
    sua_mua_classifier="SpikeInterface/UnitRefine_sua_mua_classifier",
)
print(all_labels)


##############################################################################
# If you run this without the ``trust_model=True`` parameter, you will receive an error:
#
# .. code-block::
#
#     UntrustedTypesFoundException: Untrusted types found in the file: ['sklearn.metrics._classification.balanced_accuracy_score', 'sklearn.metrics._scorer._Scorer', 'sklearn.model_selection._search_successive_halving.HalvingGridSearchCV', 'sklearn.model_selection._split.StratifiedKFold']
#
# This is a security warning, which can be overcome by passing the trusted types list
# ``trusted = ['sklearn.metrics._classification.balanced_accuracy_score', 'sklearn.metrics._scorer._Scorer', 'sklearn.model_selection._search_successive_halving.HalvingGridSearchCV', 'sklearn.model_selection._split.StratifiedKFold']``
# or by passing the ``trust_model=True``` keyword.
#
# .. dropdown:: More about security
#
#   Sharing models, which are Python objects, is complicated.
#   We have chosen to use the `skops format <https://skops.readthedocs.io/en/stable/>`_, instead
#   of the common but insecure ``.pkl`` format (read about ``pickle`` security issues
#   `here <https://lwn.net/Articles/964392/>`_). While unpacking the ``.skops`` file, each function
#   is checked. Ideally, skops should recognise all `sklearn`, `numpy` and `scipy` functions and
#   allow the object to be loaded if it only contains these (and no unkown malicious code). But
#   when ``skops`` it's not sure, it raises an error. Here, it doesn't recognise
#   ``['sklearn.metrics._classification.balanced_accuracy_score', 'sklearn.metrics._scorer._Scorer',
#   'sklearn.model_selection._search_successive_halving.HalvingGridSearchCV',
#   'sklearn.model_selection._split.StratifiedKFold']``. Taking a look, these are all functions
#   from `sklearn`, and we can happily add them to the ``trusted`` functions to load.
#
#    In general, you should be cautious when downloading ``.skops`` files and ``.pkl`` files from repos,
#    especially from unknown sources.
#
# Directly applying a sklearn Pipeline
# ------------------------------------
#
# Instead of using ``HuggingFace`` and ``skops``, someone might have given you a model
# in differet way: perhaps by e-mail or a download. If you have the model in a
# folder, you can apply it in a very similar way:
#
# .. code-block::
#
#    labels = sc.model_based_label_units(
#        sorting_analyzer = sorting_analyzer,
#        model_folder = "path/to/model/folder",
#    )

##############################################################################
# Using this, you lose the advantages of the model metadata: the quality metric parameters
# are not checked and the labels are not converted their original human readable names (like
# 'good' and 'bad'). Hence we advise using the methods discussed above, when possible.
