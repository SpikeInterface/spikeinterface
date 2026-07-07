"""
Upload a pipeline to Hugging Face Hub
=====================================
"""
##############################################################################
# In this tutorial we will upload a pipeline, trained in SpikeInterface, to the
# `Hugging Face Hub <https://huggingface.co/>`_ (HFH).
#
# To do this, you first need to train a model. `Learn how here! <https://spikeinterface.readthedocs.io/en/latest/tutorials/curation/plot_2_train_a_model.html>`_
#
# Hugging Face Hub?
# -----------------
# Hugging Face Hub (HFH) is a model sharing platform focused on AI and ML models and datasets.
# To upload your own model to HFH, you need to make an account with them.
# If you do not want to make an account, you can simply share the model folder with colleagues.
# There are also several ways to interaction with HFH: the way we propose here doesn't use
# many of the tools ``skops`` and hugging face have developed such as the ``Card`` and
# ``hub_utils``. Feel free to check those out `here <https://skops.readthedocs.io/en/stable/examples.html>`_.
#
# Prepare your model
# ------------------
#
# The plan is to make a folder with the following file structure
#
# .. code-block::
#
#     my_model_folder/
#         my_model_name.skops
#         model_info.json
#         training_data.csv
#         labels.csv
#         metadata.json
#
# SpikeInterface and HFH don't require you to keep this folder structure, we just advise it as
# best practice.
#
# If you've used SpikeInterface to train your model, the ``train_model`` function auto-generates
# most of this data. The only thing missing is the the ``metadata.json`` file. The purpose of this
# file is to detail how the model was trained, which can help prospective users decide if it
# is relevant for them. For example, taking
# a model trained on mouse data and applying it to a primate is likely a bad idea (or a
# great research paper!). And a model trained using tetrode data might have limited application
# on a silcone high-density probes. Hence we suggest saving at least the species, brain areas
# and probe information, as is done in the dictionary below. Note that we format the metadata
# so that the information
# in common with the NWB data format is consistent with it. Since the models can be trained
# on several curations, all the metadata fields are lists:
#
# .. code-block::
#
#     import json
#
#     model_metadata = {
#         "subject_species": ["Mus musculus"],
#         "brain_areas": ["CA1"],
#         "probes":
#             [{
#              "manufacturer": "IMEc",
#              "name": "Neuropixels 2.0"
#             }]
#         }
#     with open("my_model_folder/metadata.json", "w") as file:
#         json.dump(model_metadata, file)
#
# Upload to HuggingFaceHub
# ------------------------
#
# We'll now upload this folder to HFH using the web interface.
#
# First, go to https://huggingface.co/ and make an account. Once you've logged in, press
# ``+`` then ``New model`` or find ``+ New Model`` in the user menu. You will be asked
# to enter a model name, to choose a license for the model and whether the model should
# be public or private. After you have made these choices, press ``Create Model``.
#
# You should be on your model's landing page, whose header looks something like
#
# .. image:: ../../images/initial_model_screen.png
#     :width: 550
#     :align: center
#     :alt: The page shown on HuggingFaceHub when a user first initialises a model
#
# Click Files, then ``+ Add file`` then ``Upload file(s)``. You can then add your files to the repository. Upload these by pressing ``Commit changes to main``.
#
# You are returned to the Files page, which should look similar to
#
# .. image:: ../../images/files_screen.png
#     :width: 700
#     :align: center
#     :alt: The file list for a model HuggingFaceHub.
#
# Let's add some information about the model for users to see when they go on your model's
# page. Click on ``Model card`` then ``Edit model card``. Here is a sample model card for
# For a model based on synthetically generated tetrode data,
#
# .. code-block::
#
#     ---
#     license: mit
#     ---
#
#     ## Model description
#
#     A toy model, trained on toy data generated from spikeinterface.
#
#     # Intended use
#
#     Used to try out automated curation in SpikeInterface.
#
#     # How to Get Started with the Model
#
#     This can be used to automatically label a sorting in spikeinterface. Provided you have a `sorting_analyzer`, it is used as follows
#
#     ` ` ` python (NOTE: you should remove the spaces between each backtick. This is just formatting for the notebook you are reading)
#
#         from spikeinterface.curation import model_based_label_units
#         labels = model_based_label_units(
#             sorting_analyzer = sorting_analyzer,
#             repo_id = "SpikeInterface/toy_tetrode_model",
#             trust_model=True
#         )
#     ` ` `
#
#     or you can download the entire repositry to `a_folder_for_a_model`, and use
#
#     ` ` ` python
#         from spikeinterface.curation import model_based_label_units
#
#         labels = model_based_label_units(
#             sorting_analyzer = sorting_analyzer,
#             model_folder = "path/to/a_folder_for_a_model",
#             trusted = ['numpy.dtype']
#         )
#     ` ` `
#
#     # Authors
#
#     Chris Halcrow
#
# You can see the repo with this `Model card <https://huggingface.co/SpikeInterface/toy_tetrode_model>`_.
