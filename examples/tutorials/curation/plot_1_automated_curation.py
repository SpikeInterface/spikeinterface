"""
Model-based curation tutorial
=============================

This notebook provides a step-by-step guide on how to use a machine learning classifier for
curating spike sorted output. We'll download a toy model and use it to label our sorted data by using
Spikeinterface. We start by importing some packages
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
# Let's download a pretrained model from `Hugging Face <https://huggingface.co/>`_ (HF). The
# ``load_model`` function allows us to download directly from HF, or use a model in a local
# folder. The function downloads the model and saves it in a temporary folder and returns a
# model and some metadata about the model.

model, model_info = sc.load_model(
    repo_id = "SpikeInterface/toy_tetrode_model",
    trusted = ['numpy.dtype']
)


##############################################################################
# This model was trained on artifically generated tetrode data. The model object has a nice html
# representation, which will appear if you're using a Jupyter notebook.

model

##############################################################################
# The model object (an sklearn Pipeline) contains information about which metrics
# were used to compute the model. We can access it from the model (or from the model_info)

print(model.feature_names_in_)

##############################################################################
# Hence, to use this model we need to create a ``sorting_analyzer`` with all these metrics computed.
# We'll do this by generating a recording and sorting, creating a sorting analyzer and computing a
# bunch of extensions. Follow these links for more info on `recordings <https://spikeinterface.readthedocs.io/en/latest/modules/extractors.html>`_, `sortings <https://spikeinterface.readthedocs.io/en/latest/modules/sorters.html>`_, `sorting analyzers <https://spikeinterface.readthedocs.io/en/latest/tutorials/core/plot_4_sorting_analyzer.html#sphx-glr-tutorials-core-plot-4-sorting-analyzer-py>`_
# and `extensions <https://spikeinterface.readthedocs.io/en/latest/modules/postprocessing.html>`_.

recording, sorting = si.generate_ground_truth_recording(num_channels=4, seed=4, num_units=10)
sorting_analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording)
sorting_analyzer.compute(['noise_levels','random_spikes','waveforms','templates','spike_locations','spike_amplitudes','correlograms','principal_components','quality_metrics','template_metrics'])

##############################################################################
# This sorting_analyzer now contains the required quality metrics and template metrics.
# We can check that this is true by accessing the extension data.

all_metric_names = list(sorting_analyzer.get_extension('quality_metrics').get_data().keys()) + list(sorting_analyzer.get_extension('template_metrics').get_data().keys())
print(np.all(all_metric_names == model.feature_names_in_))

##############################################################################
# Great! We can now use the model to predict labels. You can either pass a HuggingFace repo, or a
# local folder containing the model file and the ``model_info.json`` file. Here, we'll pass
# a repo. The function returns a dictionary containing a label and a confidence for each unit
# contained in the ``sorting_analyzer``.

labels = sc.auto_label_units(
    sorting_analyzer = sorting_analyzer,
    repo_id = "SpikeInterface/toy_tetrode_model",
    trusted = ['numpy.dtype']
)

labels


##############################################################################
# The model has labelled one unit as bad. Let's look at that one, and the 'good' unit with the highest
# confidence of being 'good'.

sw.plot_unit_templates(sorting_analyzer, unit_ids=[7,9])

##############################################################################
# Nice - we see that unit 9 does look a lot more spikey than unit 7. You might think that unit
# 7 is a real unit. If so, this model isn't good for you.
#
# Assess the model performance
# ----------------------------
#
# To assess the performance of the model relative to human labels, we can load or generate some
# human labels, and plot a confusion matrix of predicted vs human labels for all clusters. Here
# we'll be a conservative human, who has labelled several units with small amplitudes as 'bad'.

human_labels = ['bad', 'good', 'good', 'bad', 'good', 'bad', 'good', 'bad', 'good', 'good']

# Note: if you labelled using phy, you can load the labels using:
# human_labels = sorting_analyzer.sorting.get_property('quality')

from sklearn.metrics import confusion_matrix, balanced_accuracy_score

label_conversion = model_info['label_conversion']
predictions = [ labels[a][0] for a in range(10) ]

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
# label matches the human label
#
# This could be used to set a threshold above which you accept the model's classification,
# and only manually curate those which it is less sure of.


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

confidences = sorting_analyzer.sorting.get_property('classifier_probability')

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
# In this case, you might decide to only trust labels which had confidence over above 0.86.
#
# A more realistic example: Neuropixels data
# ------------------------------------------
#
# Above, we used a toy model trained on generated data. There are also models on HuggingFace
# trained on real data.
#
# For example, the following classifier is trained on Neuropixels data from 11 mice recorded in
# V1,SC and ALM: https://huggingface.co/AnoushkaJain3/curation_machine_learning_models
#
# Go to that page, and take a look at the ``Files``. The models are contained in the
# `skops files <https://skops.readthedocs.io/en/stable/>`_ and there are *two* in this repo.
# We can choose which to load in the ``load_model`` function as follows:
#
# .. code-block::
#    import spikeinterface.curation as sc
#    model, model_info = sc.load_model(
#        repo_id = "AnoushkaJain3/curation_machine_learning_models",
#        model_name= 'noise_neuron_model.skops',
#    )
#
# If you run this locally you will receive an error:
#
# .. code-block::
#
#     UntrustedTypesFoundException: Untrusted types found in the file: ['sklearn.metrics._classification.balanced_accuracy_score', 'sklearn.metrics._scorer._Scorer', 'sklearn.model_selection._search_successive_halving.HalvingGridSearchCV', 'sklearn.model_selection._split.StratifiedKFold']
#
# This is a security warning. Sharing models, with are Python objects, is complicated.
# We have chosen to use the `skops format <https://skops.readthedocs.io/en/stable/>`_, instead
# of the common but insecure ``.pkl`` format (read about ``pickle`` security issues
# `here <https://lwn.net/Articles/964392/>`_). While unpacking the ``.skops`` file, each function
# is checked. Ideally, skops should recognise all `sklearn`, `numpy` and `scipy` functions and
# allow the object to be loaded if it only contains these (and no unkown malicious code). But
# when ``skops`` it's not sure, it raises an error. Here, it doesn't recognise
# ``['sklearn.metrics._classification.balanced_accuracy_score', 'sklearn.metrics._scorer._Scorer',
# 'sklearn.model_selection._search_successive_halving.HalvingGridSearchCV',
# 'sklearn.model_selection._split.StratifiedKFold']``. Taking a look, these are all functions
# from `sklearn`, and we can happily add them to the ``trusted`` functions to load:
#
# .. code-block::
#
#     model, model_info = sc.load_model(
#         model_name = 'noise_neuron_model.skops',
#         repo_id = "AnoushkaJain3/curation_machine_learning_models",
#         trusted = ['sklearn.metrics._classification.balanced_accuracy_score', 'sklearn.metrics._scorer._Scorer', 'sklearn.model_selection._search_successive_halving.HalvingGridSearchCV', 'sklearn.model_selection._split.StratifiedKFold']
#     )
#
# As ``skops`` continues to be developed, we hope more of these functions will be :code:`trusted`
# by default.
#
# If you unequivocally trust the model (e.g. if you have created it), you can bypass this security
# step by passing ``trust_model = True`` to the ``load_model`` function.
#
# In general, you should be cautious when downloading ``.skops`` files and ``.pkl`` files from repos,
# especially from unknown sources.
#
#
# Directly applying a sklearn Pipeline
# ------------------------------------
#
# Instead of using ``HuggingFace`` and ``skops``, you might have another way of receiving a sklearn
# pipeline, and want to apply it to your sorted data.

from spikeinterface.curation.model_based_curation import ModelBasedClassification

model_based_classification = ModelBasedClassification(sorting_analyzer, model)
labels = model_based_classification.predict_labels()
labels

##############################################################################
# Using this, you lose the advantages of the model metadata: the quality metric parameters
# are not checked and the labels are not converted their original human readable names (like
# 'good' and 'bad'). Hence we advise using the methods discussed above, when possible.
