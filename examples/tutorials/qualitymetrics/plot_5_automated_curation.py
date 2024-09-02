"""
Model Based Curation Tutorial
=============================

This notebook outlines approaches to training a machine learning classifier on Spikeinterface-computed quality metrics, and using such models to predict curation labels for previously uncurated electrophysiology data
  - Code for predicting labels using a trained model can be found in the first section, then code for training your own bespoke model
  - Plots can be generated to assess the performance of the model, both on the training and unseen data
  - Pre-trained models can be downloaded from `Hugging Face <https://huggingface.co/>`_, or opened from `skops <https://skops.readthedocs.io/en/stable/>`_ files
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.curation.model_based_curation import compute_all_metrics

from os import cpu_count

# Set the number of CPU cores to be used globally - defaults to all cores -1
n_cores = cpu_count() -1
si.set_global_job_kwargs(n_jobs = n_cores)
print(f"Number of cores set to: {n_cores}")

# SET OUTPUT FOLDER
output_folder = "/home/jake/Documents/ephys_analysis/code/ephys_analysis/auto_curation/models"

##############################################################################
# Applying a pretrained model to predict curation labels
# ------------------------------
# 
# We supply pretrained machine learning classifiers for predicting spike-sorted clusters with arbitrary labels, in this example single-unit activity ('good'), or noise. This particular approach works as follows:
# 
#   1. Create a Spikeinterface 'sorting analyzer <https://spikeinterface.readthedocs.io/en/latest/modules/core.html#sortinganalyzer>`_ and compute `quality metrics <https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html>`_
#
#   2. Load a pretrained classification model from Hugging Face & skops
#
#   3. Compare with human-applied curation labels to assess performance, optionally export labels to phy to allow manual checking
# 
# Load data and compute quality metrics - test data are used here, but replace with your own recordings!
# We would recommend to start with some previously labelled data, to allow comparison of the model performance against your labelling
# First, let's simulate some data and compute quality metrics

unlabelled_recording, unlabelled_sorting = si.generate_ground_truth_recording(durations=[60], num_units=30)

unlabelled_analyzer = si.create_sorting_analyzer(sorting = unlabelled_sorting, recording = unlabelled_recording, sparse = True)

# Compute all quality metrics
quality_metrics, template_metrics = compute_all_metrics(unlabelled_analyzer)

##############################################################################
# Load pretrained models and predict curation labels for unlabelled data
# ------------------------------
# 
# We can download a pretrained model from Hugging Face, and use this to test out prediction. This particular model assumes only two labels ('noise', and 'good') have been used for classification
# Predictions and prediction confidence are then stored in "label_prediction" and "label_confidence" unit properties

##############################################################################
# Load pretrained noise/neural activity model and predict on unlabelled data
from spikeinterface.curation.model_based_curation import auto_label_units

from huggingface_hub import hf_hub_download
import skops.io
model_path = hf_hub_download(repo_id="chrishalcrow/test_automated_curation_3", filename="skops-_xvuw15v.skops")
model = skops.io.load(model_path, trusted='numpy.dtype')

label_conversion = {0: 'noise', 1: 'good'}

label_dict = auto_label_units(sorting_analyzer=unlabelled_analyzer,
                              pipeline=model,
                              label_conversion=label_conversion,
                              export_to_phy=False,
                              pipeline_info_path=None)
unlabelled_analyzer.sorting

##############################################################################
# Assess model performance by comparing with human labels

# To assess the performance of the model relative to human labels, we can load (or here generate randomly) some labels, and plot a confusion matrix of predicted vs human labels for all clusters

from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import seaborn as sns

# Use 'ground-truth' labels to check prediction accuracy
# These are assigned randomly here but you could load these from phy 'cluster_group.tsv', from the 'quality' property of the sorting, or similar
human_labels = np.random.choice(list(label_conversion.values()), unlabelled_analyzer.get_num_units())

# Get labels from phy sorting (if loaded) using:
# human_labels = unlabelled_analyzer.sorting.get_property('quality')

predictions = unlabelled_analyzer.sorting.get_property('label_prediction')

conf_matrix = confusion_matrix(human_labels, predictions)

# Calculate balanced accuracy for the confusion matrix
balanced_accuracy = balanced_accuracy_score(human_labels, predictions)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis')
plt.xlabel('Predicted Label')
plt.ylabel('Human Label')
plt.xticks(ticks = [0.5, 1.5], labels = list(label_conversion.values()))
plt.yticks(ticks = [0.5, 1.5], labels = list(label_conversion.values()))
plt.title('Predicted vs Human Label')
plt.suptitle(f"Balanced Accuracy: {balanced_accuracy}")
plt.show()

##############################################################################
# We can also see how the model's confidence relates to the probability that the model label matches the human label
# 
# This could be used to set a threshold above which you might accept the model's classification, and only manually curate those which it is less sure of

confidences = unlabelled_analyzer.sorting.get_property('label_confidence')

# Make dataframe of human label, model label, and confidence
label_df = pd.DataFrame(data = {
    'phy_label': human_labels,
    'decoder_label': predictions,
    'confidence': confidences},
    index = unlabelled_analyzer.sorting.get_unit_ids())

# Calculate the proportion of agreed labels by confidence decile
label_df['model_x_human_agreement'] = label_df['phy_label'] == label_df['decoder_label']

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

p_agreement_sorted, p_agreement_moving_avg = calculate_moving_avg(label_df, 'confidence', 20)

# Plot the moving average of agreement
plt.figure(figsize=(6, 6))
plt.plot(p_agreement_sorted, p_agreement_moving_avg, label = 'Moving Average')
plt.axhline(y=1/len(np.unique(predictions)), color='black', linestyle='--', label='Chance')
plt.xlabel('Confidence'); plt.xlim(0.5, 1)
plt.ylabel('Proportion Agreement with Human Label'); plt.ylim(0, 1)
plt.title('Agreement vs Confidence (Moving Average)')
plt.legend(); plt.grid(True); plt.show()

##############################################################################
# ------------------------------
# 
##############################################################################
# Training a new model
# 
# **If the pretrained models do not give satisfactory performance on your data, it is easy to train your own classifier through SpikeInterface!**
# 
# First we make a Spikeinterface SortingAnalyzer object, in this case using simulated data, and generate some labels for the units to use as our target
# Note that these labels are random as written here, so the example model will only perform at chance
# 
# Load some of your data and curation labels here and see how it performs!
# **Note that it is likely that for useful generalisability, you will need to use multiple labelled recordings for training.** To do this, compute metrics as described for multiple SortingAnalyzers, then pass them as a list to the model training function, and pass the labels as a single list in the same order
# 
# The set of unique labels used is arbitrary, so this could just as easily be used for any cluster categorisation task as for curation into the standard 'good', 'mua' and 'noise' categories

# Make a simulated SortingAnalyzer with 100 units
labelled_recording, labelled_sorting = si.generate_ground_truth_recording(durations=[60], num_units=30)

labelled_analyzer = si.create_sorting_analyzer(sorting = labelled_sorting, recording = labelled_recording, sparse = True)

# Compute all quality metrics
compute_all_metrics(labelled_analyzer)

label_conversion = {'noise': 0, 'mua': 1, 'good': 2}

# These are assigned randomly here but you could load these from phy 'cluster_group.tsv', from the 'quality' property of the sorting, or similar
human_labels = np.random.choice(list(label_conversion.values()), labelled_analyzer.get_num_units())
labelled_analyzer.sorting.set_property('quality', human_labels)

# Get labels from phy sorting (if loaded) using:
# human_labels = unlabelled_analyzer.sorting.get_property('quality')

##############################################################################
# Now we train the machine learning classifier
# 
# By default, this searches a range of possible imputing and scaling strategies, and uses a Random Forest classifier. It then selects the model which most accurately predicts the supplied 'ground-truth' labels
# 
# As output, this function saves the best model (as a skops file, similar to a pickle), a csv file containing information about the performance of all tested models (`model_label_accuracies.csv`), and a `model_info.json` file containing the parameters used to compute quality metrics, and the SpikeInterface version, for reproducibility

# Load labelled metrics and train model
from spikeinterface.curation.train_manual_curation import train_model

# We will use a list of two (identical) analyzers here, we would advise using more than one to improve model performance
trainer = train_model(mode = "analyzers",
    labels = np.append(human_labels, human_labels),
    analyzers = [labelled_analyzer, labelled_analyzer],
    output_folder = output_folder, # Optional, can be set to save the model and model_info.json file
    metric_names = None, # Can be set to specify which metrics to use for training
    imputation_strategies = None, # Default to all
    scaling_techniques = None, # Default to all
    classifiers = None, # Default to Random Forest only
    seed = None)

best_model = trainer.best_pipeline
best_model

# OR load model from file
# import skops.io
# pipeline_path = Path(output_folder) / Path("best_model_label.skops")
# unknown_types = skops.io.get_untrusted_types(file=pipeline_path)
# best_model = skops.io.load(pipeline_path, trusted=unknown_types)

##############################################################################
# We can see the performance of each model in this `model_label_accuracies.csv` output file

# Load and disply top 5 pipelines and accuracies
accuracies = pd.read_csv(Path(output_folder) / Path("model_label_accuracies.csv"), index_col = 0)
accuracies.head()

##############################################################################
# We can also see which metrics are most important to our model:

# Plot feature importances
importances = best_model.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]
features = best_model.feature_names_in_
n_features = best_model.n_features_in_

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(n_features), importances[indices], align="center")
plt.xticks(range(n_features), features, rotation=90)
plt.xlim([-1, n_features])
plt.show()

##############################################################################
# Apply trained model to unlabelled data 
# 
# This approach is the same as in the first section, but without the need to combine the output of two separate classifiers

unlabelled_recording, unlabelled_sorting = si.generate_ground_truth_recording(durations=[60], num_units=30)
unlabelled_analyzer = si.create_sorting_analyzer(sorting = unlabelled_sorting, recording = unlabelled_recording, sparse = True)

compute_all_metrics(unlabelled_analyzer)

##############################################################################
# Load best model and predict on unlabelled data

from spikeinterface.curation.model_based_curation import auto_label_units
label_conversion = {0: 'noise', 1: 'mua', 2: 'good'}
label_dict = auto_label_units(sorting_analyzer = unlabelled_analyzer,
                              pipeline = best_model,
                              label_conversion = label_conversion,
                              export_to_phy = False,
                              pipeline_info_path = Path(output_folder) / Path("model_info.json"))
unlabelled_analyzer.sorting