
# From WaveformExtractor to SortingAnalyzer

From spikeinterface version 0.101, the internal structure of the postprocessing module has
changed blah blah. This change should allow for faster blah blah and more flexible blah blah blah and
better provenance, as the new changes tightly couple recordings and sortings. We
explain the motivation for the change in more detail [below](#why-change).

From the user point of view, the key change is the deletion of the `WaveformExtractor` class and the addition of
the new `SortingAnalyzer` class. Hence any code using `WaveformExtractors` will have to be updated.
If you want to continue using `WaveformExtractor` you need to use spikeinterface versions 0.100 and
below.

Updating your old code should be straightforward. On this page, we demonstrate how to [convert old WaveformExtractor folders
to new SortingAnalyzer folders](#convert-a-waveformextractor-folder-to-a-sortinganalyzer-folder) and have written a 
[code dictionary](#dictionary-between-sortinganalyzer-and-waveformextractor) which should make updating your codebase simple.

## Why change?

The problems with WaveformExtractor...

The advantages of SortingAnalyzer...


## Convert a WaveformExtractor folder to a SortingAnalyzer folder

There are several backward compatibility tools to help deal with existing `WaveformExtractor` folders.
In the short term, you can still read these folders using `load_waveforms`, which creates a `MockWaveformExtractor` object.
This object contains a `SortingAnalyzer` which can be accessed and then saved as follows

```
waveform_folder_path = "path_to/my_waveform_extractor_folder"
new_sorting_analyzer_path = "path_to/my_new_sorting_analyzer_folder"

extractor = load_waveform(folder = waveform_folder_path)
sorting_analyzer = extractor.sorting_analyzer
sorting_analyzer.save_as(folder = new_sorting_analyzer_path, format = "binary_folder")
```

The above code creates a `SortingAnalyzer` folder at `new_sorting_analyzer_path`.

## Dictionary between SortingAnalyzer and WaveformExtractor 

This section provides a dictionary for code, to help translate between a `SortingAnalyzer`
and a `WaveformExtractor`, so that users can easily update old code. If you want to learn
how to use `SortingAnalyzer` from scratch, the 
[Get Started](https://spikeinterface.readthedocs.io/en/latest/how_to/get_started.html) guide
and the [PostProcessing module documentation](https://spikeinterface.readthedocs.io/en/latest/modules/postprocessing.html) 
are better places to start.

This section is split into four subsections:
- [Create, load and save](#create-load-and-save)
- [Checking basic properties](#checking-basic-properties)
- [Compute Extensions](#compute-extensions)
- [Quality Metrics](#quality-metrics)

Throughout this section, we assume that all functions have been imported into the namespace. 
E.g. we have run code like `from spikeinterface.postprocessing import extract_waveforms` for each function. If you have imported
the full package (ie you have run `import spikinterface.full as si`) you need to prepend all
functions by `si.`. If you have imported individual modules (ie you have run `import spikinterface.postprocessing as spost`)
we will need to prepend functions by the appropriate submodule name.


In the following we start with a recording called `recording` and a sorting
object called `sorting`. We’ll then create, export, explore and compute things using a
`SortingAnalyzer` called `analyzer` and a`WaveformExtractor` called `extractor`. 
We’ll do the same calculations for both objects. The WaveformExtractor code will be on
the left while the SortingAnalyzer code will be displayed on the right:

::::{grid} 2
:::{grid-item} 
# WaveformExtractor
:::
:::{grid-item} 
# SortingAnalyzer
:::
::::

### Create, load and save

First, create the object from a recording and a sorting. You can save a copy of the
`SortingAnalyzer` locally by specifying a `folder` and a `format`.

::::{grid} 2
:::{grid-item}

```
extractor = extract_waveforms(
    sorting = sorting, 
    recording = recording)
```
:::
:::{grid-item}
```
analyzer = create_sorting_analyzer(
    sorting = sorter, 
    recording = recording)
```
:::
::::

By default, the object is stored in memory. Alternatively, we can save it locally 
at the point of creation by specifying a `folder` and a `format`. Additionally,
you can decide whether to use sparsity or not at this point

::::{grid} 2
:::{grid-item}
```
extractor = extract_waveforms(
    sorting = sorter, 
    recording = recording,
    mode = "folder",
    folder = "my_waveform_extractor",
    sparse = True)
```
:::
:::{grid-item}
```
analyzer = create_sorting_analyzer(
    sorting = sorter, 
    recording = recording,
    folder = "my_sorting_analyzer",
    format = "binary_folder",
    sparse = True)
```
:::
::::

Or you can save the object after you've created it, with the option
of saving it to a new format

::::{grid} 2
:::{grid-item}
```
extractor.save(format="zarr", 
    folder="/path/to_my/result.zarr")
```
:::
:::{grid-item}
```
analyzer.save_as(format="zarr", 
    folder="/path/to_my/result.zarr")
```
:::
::::

If you already have the object saved, you can load it

::::{grid} 2
:::{grid-item}
```
extractor = load_waveforms(
    folder="my_waveform_extractor")
```
:::
:::{grid-item}
```
analyzer = load_sorting_analyzer(
    folder="my_sorting_analyzer")
```
:::
::::

### Checking basic properties

The object contains both a `sorting` and a `recording` object. These
can be isolated


::::{grid} 2
:::{grid-item}
```
the_recording = extractor.recording
the_sorting = extractor.sorting

```
:::
:::{grid-item}
```
the_recording = analyzer.recording
the_sorting = analyzer.sorting
```
:::
::::

You can then check any `recording` or `sorting` properties from these objects.

There is much information about the recording and sorting contained in the parent object. E.g. you can get
the channel locations as follows

::::{grid} 2
:::{grid-item}
```
channel_locations = 
    extractor.get_channel_locations()
```
:::
:::{grid-item}
```
channel_locations = 
    analyzer.get_channel_locations()
```
:::
::::

Many properties can be accessed in a similar way

::::{grid} 2
:::{grid-item}
```
extractor.get_num_channels()
extractor.get_num_samples()
extractor.get_num_segments()
extractor.get_probe()
extractor.get_probegroup()
extractor.get_total_duration()
extractor.get_total_samples()
```
:::
:::{grid-item}
```
analyzer.get_num_channels()
analyzer.get_num_samples()
analyzer.get_num_segments()
analyzer.get_probe()
analyzer.get_probegroup()
analyzer.get_total_duration()
analyzer.get_total_samples()
```
:::
::::

...while some are simply properties of the object

::::{grid} 2
:::{grid-item}
```
extractor.channel_ids
extractor.unit_ids
extractor.sampling_frequency
```
:::
:::{grid-item}
```
analyzer.channel_ids
analyzer.unit_ids
analyzer.sampling_frequency
```
:::
::::

You can also find some fundamental properties of the object,
though these are mostly used internally:

::::{grid} 2
:::{grid-item}
```
extractor.folder
extractor.format
extractor.is_read_only()
extractor.dtype
extractor.is_sparse()
```
:::
:::{grid-item}
```
analyzer.folder
analyzer.format
analyzer.is_read_only()
analyzer.get_dtype()
analyzer.is_sparse()
```
:::
::::

### Compute Extensions

Waveforms, templates, quality metrics etc are all extensions of the `SortingAnalyzer` object.
Some extensions depend on other extensions. To calculate a _parent_ we must first have calculated it's 
_children_. The relationship between some commonly used extensions are shown below:

![Child parent relationships](waveform_extractor_to_sorting_analyzer_files/child_parent_plot.svg)

We see that to compute `spike_amplitudes` we must first compute `templates`. To compute templates
we must first compute `waveforms`. To compuate waveforms we must first compute `random_spikes`. Phew.
Some of these extensions were calulcated automatically for WaveformExtractors, so the code
looks slightly different. Let's calculate these extensions, and also add a parameter for `spike_amplitudes`

::::{grid} 2
:::{grid-item}
```
  
extractor.precompute_templates(
   modes=("average",))
compute_spike_amplitudes(extractor,
    peak_sign = "pos")
```
:::
:::{grid-item}
```
analyzer.compute("random_spikes")
analyzer.compute("waveforms")
analyzer.compute("templates")
analyzer.compute("spike_amplitudes",
    peak_sign = "pos")
```
:::
::::

Read more about extensions and their keyword arguments in the
[PostProcessing module documentation](https://spikeinterface.readthedocs.io/en/latest/modules/postprocessing.html.

In many cases, you can still use the old notation for `SortingAnalyzer` objects,
such as `compute_spike_amplitudes(analyzer=analyzer)`.

In all cases, if the object has been saved locally, the extensions will be saved
locally too. You can check which extensions have been saved

::::{grid} 2
:::{grid-item}
```
extractor.get_available_extension_names()
```
:::
:::{grid-item}
```
analyzer.get_saved_extension_names()
```
:::
::::

You can now also check which extensions are currently loaded _in memory_. The WaveformExtractor
checks the local folder _and_ the memory:

::::{grid} 2
:::{grid-item}
```
extractor.get_available_extension_names()
```
:::
:::{grid-item}
```
analyzer.get_loaded_extension_names()
```
:::
::::

If there is an extensions which is saved but not yet loaded you can load it:

::::{grid} 2
:::{grid-item}
```
extractor.load_extension(
    extension_name = "spike_amplitudes")
```
:::
:::{grid-item}
```
analyzer.load_extension(
    extension_name = "spike_amplitudes")
```
:::
::::

You can also check if a certain extension is loaded

::::{grid} 2
:::{grid-item}
```
extractor.has_extension(
    extension_name = "spike_amplitudes")
```
:::
:::{grid-item}
```
analyzer.has_extension(
    extension_name = "spike_amplitudes")
```
:::
::::

You can delete extensions. Note that if you delete a child all of its parents
will be deleted too. You cannot delete `templates` from a WaveformExtractor,
so we'll delete `spike_amplitudes` instead.

::::{grid} 2
:::{grid-item}
```
  

extractor.delete_extension(
    extension_name = "spike_amplitudes")
```
:::
:::{grid-item}
```
# This also deletes any parents
# such as spike_amplitudes
analyzer.delete_extension(
    extension_name = "templates")
```
:::
::::

Once you have computed an extension, you often want to look at the data associated with it.
This has been standardised for the `SortingAnalyzer` object, through the `get_data` method.
The retrieval methods for the `WaveformExtractor` object were less uniform, and depended
on which extension you were interested in. We won't list them all here.

::::{grid} 2
:::{grid-item}
```
  
wv_data = extractor.get_waveforms(
    unit_id=0)
  
ul_data = compute_unit_locations(
    extractor)
```
:::
:::{grid-item}
```
wv = analyzer.get_extension(
    extension_name = "waveforms")
wv_data = wv.get_data()
ul = analyzer.get_extension(
    extension_name = "unit_locations")
ul_data = nl.get_data()
```
:::
::::

You can also access the parameters used in the extension calculation. The WaveformExtractor does not have
as nice a method...

::::{grid} 2
:::{grid-item}
```
ul_ext = extractor.load_extension(
    "unit_locations")
ul_parms = ul_ext.load_params_from_folder(
    folder="my_waveform_extractor")
```
:::
:::{grid-item}
```
  
  
ul_parms = ul.params
```
:::
::::

### Quality metrics

Quality metrics for the `SortingAnalyzer` are just extensions. You can calculate a specific
quality metric using the `metric_names` argument. In contrast, for `WaveformExtractor`s you 
need to find the correct function. The old functions still work for `SortingAnalyzer`s.

::::{grid} 2
:::{grid-item}
```
amp_cut_data = compute_amplitude_cutoffs(
    waveform_extractor = extractor)
  
  
#or: compute_amplitude_cutoffs(extractor)
```
:::
:::{grid-item}
```
amp_cutoff = analyzer.compute(
    "quality_metrics", 
    metric_names=["amplitude_cutoff"])
amp_cut_data = amp_cutoff.get_data()
#or: compute_amplitude_cutoff(analyzer)

```
:::
::::

Or you can calculate all available quality metrircs. You might want to pass a 
list of quality metric parameters too.

::::{grid} 2
:::{grid-item}
```
dqm_params = get_default_qm_params()
amp_cut_data = compute_quality_metrics(
    waveform_extractor = extractor,
    qm_params = dqm_params)
```
:::
:::{grid-item}
```
dqm_params = get_default_qm_params()
amp_cutoff = analyzer.compute(
    "quality_metrics",
    qm_params = dqm_params)
#alt: compute_quality_metrics(analyzer)
```
:::
::::

Learn more about the possible quality metrics and their keyword arguments in the
[quality metrics documentation page](https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html).
