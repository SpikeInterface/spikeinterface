# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# # Build a full Sorting pipeline with dicts
#
# When using `SpikeInterface` there are two phases. First, you should play: try to figure out
# any special steps or parameters you need to play with to get everything working with your
# data. Once your happy, you then need to build a sturdy, consistent pipeline to process all
# your ephys sessions.

# It is now possible to create a flexible spike sorting pipeline using three simple dictionaries:
# one for preprocessing (and the `PreprocessingPipeline`), another for sorting (and `run_sorter`),
# and a final one for postprocessing (and the `compute` method). Here's an example:

# +
import spikeinterface.full as si

my_protocol = {
    'preprocessing': {
        'bandpass_filter': {},
        'common_reference': {'operator': 'average'},
        'detect_and_remove_bad_channels': {},
    },
    'sorting': {
        'sorter_name': 'mountainsort5',
        'verbose': False,
        'snippet_T2': 15,
        'remove_existing_folder': True,
        'progress_bar': False
    },
    'postprocessing': {
        'random_spikes': {},
        'noise_levels': {},
        'templates': {},
        'unit_locations': {'method': 'center_of_mass'},
        'spike_amplitudes': {},
        'correlograms': {},
    },
}

# Usually, you would read in your raw recording
rec, _ = si.generate_ground_truth_recording(num_channels=4, durations=[60], seed=0)
preprocessed_rec = si.apply_pipeline(rec, my_protocol['preprocessing'])
sorting = si.run_sorter(recording=preprocessed_rec, **my_protocol['sorting'])
analyzer = si.create_sorting_analyzer(recording=preprocessed_rec, sorting=sorting)
analyzer.compute(my_protocol['postprocessing'])
# -

# This is a full and flexible spike sorting pipeline in 5 lines of code!

# To try out a different pipeline, you only need to update your protocol dicts.

# Once you have an analyzer, you can then do things with it:

analyzer.save_as(folder="my_analyzer")
si.plot_unit_summary(analyzer, unit_id=1)


# The main disadvantage of the dictionaties approach is that you don't know exactly what options
# and steps are available for you. You can search the API for help. Or we store many dictionaries 
# of tools and parameters, as is shown below.

# Get all preprocessing steps:

from spikeinterface.preprocessing.pipeline import pp_names_to_functions
print(pp_names_to_functions.keys())


# You can then check the arguments of each preprocessing step using e.g. their docstrings
# (in Jupyter you can run `si.bandpass_filter?` and in the terminal `help(si.bandpass_fitler)`)

print(si.bandpass_filter.__doc__)

# Get the default sorter parameters of mountainsort5:

print(si.get_default_sorter_params('mountainsort5'))

# Find the possible extensions you can compute

print(analyzer.get_computable_extensions())

# And the arguments for each extension 'blah' can be found in the docstring of 'compute_blah', e.g.

print(si.compute_spike_amplitudes.__doc__)
