"""
Training a model for automated curation
=============================

If the pretrained models do not give satisfactory performance on your data, it is easy to train your own classifier using SpikeInterface.
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import spikeinterface.full as si
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Note, you can set the number of cores you use using e.g.
# si.set_global_job_kwargs(n_jobs = 8)

##############################################################################
# Step 1: Generate and label data
# -------------------------------
#
# We supply pretrained machine learning classifiers for predicting spike-sorted clusters with
# arbitrary labels, in this example single-unit activity ('good'), or noise. This particular
# approach works as follows:
#
# For the tutorial, we will use simulated data to create :code:`recording` and :code:`sorting` objects. We'll
# create two sorting objects: :code:`sorting_1` is coupled to the real recording, so will contain good
# units, :code:`sorting_2` is uncoupled, so should be pure noise. We'll combine the two into one sorting
# object using :code:`si.aggregate_units`.
#
# You should `load your own recording <https://spikeinterface.readthedocs.io/en/latest/modules/extractors.html>`_ and `do a sorting <https://spikeinterface.readthedocs.io/en/latest/modules/sorters.html>`_ on your data.

recording, sorting_1 = si.generate_ground_truth_recording(num_channels=4, seed=1, num_units=5)
_, sorting_2 =si.generate_ground_truth_recording(num_channels=4, seed=2, num_units=5)

both_sortings = si.aggregate_units([sorting_1, sorting_2])

##############################################################################
# Our model is based on :code:`quality_metrics`, which are computed using a :code:`sorting_analyzer`. So we'll
# now create a sorting analyzer and compute all the extensions needed to get the quality metrics.

analyzer = si.create_sorting_analyzer(sorting = both_sortings, recording=recording)
analyzer.compute(['noise_levels','random_spikes','waveforms','templates','spike_locations','spike_amplitudes','correlograms','principal_components','quality_metrics','template_metrics'])

##############################################################################
# Let's plot the templates for the first and fifth units. The first (unit id 0) belonged to
# :code:`sorting_1` so should look like a real unit; the sixth (unit id 5) belonged to :code:`sorting_2`
# so should look like noise.

si.plot_unit_templates(analyzer, unit_ids=[0,5])

##############################################################################
# This is as expected: great! Find out more about plotting using widgets `here <https://spikeinterface.readthedocs.io/en/latest/modules/widgets.html>`_. The labels
# for our units are then easy to put in a list:

labels = ['good', 'good', 'good', 'good', 'good', 'bad', 'bad', 'bad', 'bad', 'bad']

##############################################################################
# Step 2: Train our model
# -----------------------

# With our labelled data in hand, we can train the model using the :code:`train_model` function.
# Here, the idea is that the trainer will try several classifiers, imputation strategies and
# scaling techniques then save the most accurate. To save time, we'll only try one classifier
# (Random Forest), imputation strategy (median) and scaling technique (standard scaler).

output_folder = "my_model"

# We will use a list of one analyzer here, we would strongly advise using more than one to
# improve model performance
trainer = si.train_model(
    mode = "analyzers", # You can supply a labelled csv file instead of an analyzer
    labels = [labels],
    analyzers = [analyzer],
    output_folder = output_folder, # Where to save the model and model_info.json file
    metric_names = None, # Specify which metrics to use for training: by default uses those already calculted
    imputation_strategies = ["median"], # Defaults to all
    scaling_techniques = ["standard_scaler"], # Defaults to all
    classifiers = None, # Default to Random Forest only. Other classifiers you can try [ "AdaBoostClassifier","GradientBoostingClassifier","LogisticRegression","MLPClassifier"]
)

best_model = trainer.best_pipeline

##############################################################################
#
# The above code saves the model in :code:`model.skops`, some metadata in :code:`model_info.json` and
# the model accuracies in :code:`model_accuracies.csv` in the specified :code:`output_folder`.
#
# :code:`skops` is a file format; you can think of it as a more-secture pkl file. `Read more <https://skops.readthedocs.io/en/stable/index.html>`_.
#
# The :code:`model_accuracies.csv` file contains the accuracy, precision and recall of the tested models.
# Let's take a look

accuracies = pd.read_csv(Path(output_folder) / "model_accuracies.csv", index_col = 0)
accuracies.head()

# Our model is perfect!! This is because the task was _very_ easy. We had 10 units; where
# half were pure noise and half were not.
#
# The model also contains some more information, such as which features are importantly.
# We can plot these as follows:

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
