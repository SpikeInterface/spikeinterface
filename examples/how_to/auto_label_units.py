# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Automatic labeling units after spike sorting
#
# This example shows how to automatically label units after spike sorting, using three different approaches:
#
# 1. Simple filter based on quality metrics
# 2. Bombcell: heuristic approach to label units based on quality and template metrics [Fabre]_
# 3. UnitRefine: pre-trained classifiers to label units as noise or SUA/MUA [Jain]_

# %%
import spikeinterface as si
import spikeinterface.curation as sc
import spikeinterface.widgets as sw

# %%
analyzer_path = "/ssd980/working/analyzer_np2_shank1.zarr"

# %%
analyzer = si.load(analyzer_path)

# %%
qm = analyzer_zarr.compute("quality_metrics", delete_existing_metrics=True)
qm.get_data()

# %%
