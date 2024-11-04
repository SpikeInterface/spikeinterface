"""
Upload a pipeline to Hugging Face Hub
=====================================
"""
##############################################################################
# In this tutorial we will upload a pipeline, trained in SpikeInterface, to the
# `Hugging Face Hub <https://huggingface.co/>`_ (HFH). To do this, you need a HFH account.
# If you do not want to make an account, you can simply share the model folder with colleagues.
# There are also several ways to interaction with HFH: the way we propose here doesn't use
# many of the tools ``skops`` and hugging face have developed such as the ``Card`` and
# ``hub_utils``. Feel free to check those out `here <https://skops.readthedocs.io/en/stable/examples.html>`_.
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
# SpikeInterface doesn't require you to keep this folder structure, we just advise it as
# best practice.
#
# If you've used SpikeInterface to train your model, you have already created such a folder,
# containing everything except the ``metadata.json`` file. In this file, we suggest saving
# any information which shows when a model is applicable (and when it is *not*). Taking
# a model trained on mouse data and applying it to a primate is likely a bad idea (or a
# great research paper!). And a model trained in tetrode data will have limited application
# on a silcone high-density probe. Hence we suggest the following dictionary as a minimal
# amount of information needed. Note that we format the metadata so that the information
# in common with the NWB data format is consistent with it. Since the models can be trained
# on several curations, all the metadata fields are lists:
#
# .. code-block::
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
#
#     import json
#     with open("my_model_folder/metadata.json", "w") as file:
#         json.dump(model_metadata, file)
#
# You could now share this folder with a colleague, or upload it to github. Or if you'd
# like to upload the model to Hugging Face Hub, keep reading. We'll use the
# HFH web interface.
#
# First, go to https://huggingface.co/ and make an account. Once you've logged in, press
# ``+`` then ``New model`` or find ``+ New Model`` in the user menu. You will be asked
# to enter a model name, to choose a license for the model and whether the model should
#  be public or private. After you have made these choices, press ``Create Model``.
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
#         from spikeinterface.curation import auto_label_units
#         labels = auto_label_units(
#             sorting_analyzer = sorting_analyzer,
#             repo_id = "SpikeInterface/toy_tetrode_model",
#             trust_model=True
#         )
#     ` ` `
#
#     or you can download the entire repositry to `a_folder_for_a_model`, and use
#
#     ` ` ` python
#         from spikeinterface.curation import auto_label_units
#
#         labels = auto_label_units(
#             sorting_analyzer = sorting_analyzer,
#             model_folder = "SpikeInterface/a_folder_for_a_model",
#             trusted = ['numpy.dtype']
#         )
#     ` ` `
#
#     # Authors
#
#     Chris Halcrow
#
# You can see the repo with this Model card `here <https://huggingface.co/SpikeInterface/toy_tetrode_model>`_.
