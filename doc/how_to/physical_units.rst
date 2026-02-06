
.. _physical_units:

Work with physical units in SpikeInterface recordings
=====================================================

In neurophysiology recordings, data is often stored in raw ADC (Analog-to-Digital Converter) integer values but needs to be analyzed in physical units.
For extracellular recordings, this is typically microvolts (µV), but some recording devices may use different physical units.
SpikeInterface provides tools to handle both situations.

It's important to note that **most spike sorters work fine on raw digital (ADC) units** and scaling is not needed. Going a step further, some sorters, such as Kilosort 3, require their input to be in raw ADC units.
The specific behavior however depends on the spike sorter, so it's important to understand the specific input requirements on a case per case basis.

Many preprocessing tools are also linear transformations, and if the ADC is implemented as a linear transformation which is fairly common, then the overall effect can be preserved.
That is, **preprocessing steps can often be applied either before or after unit conversion without affecting the outcome.**. That being said, there are rough edges to this approach.
preprocessing algorithms like filtering, whitening, centering, interpolation and common reference require casting to float within the pipeline. We advise users to experiment
with different approaches to find the best one for their specific use case.


Therefore, **it is usually safe to work in raw ADC integer values unless a specific tool or analysis requires physical units**.
If you are interested in visualizations, comparability across devices, or outputs with interpretable physical scales (e.g., microvolts), converting to physical units is recommended.
Otherwise, remaining in raw units can simplify processing and preserve performance.

Understanding Physical Units
----------------------------

Most recording devices store data in ADC units (integers) to save space and preserve the raw data.
To convert these values to physical units, two parameters are needed:

* **gain**: A multiplicative factor to scale the raw values
* **offset**: An additive factor to shift the values

The conversion formula is:

.. code-block:: text

    physical_value = raw_value * gain + offset


Converting to Physical Units
----------------------------

SpikeInterface provides two preprocessing classes for converting recordings to physical units. Both wrap the
``RecordingExtractor`` class and ensures that the data is returned in physical units when calling `get_traces <https://spikeinterface.readthedocs.io/en/stable/api.html#spikeinterface.core.BaseRecording.get_traces>`_

1. ``scale_to_uV``: The primary function for extracellular recordings. SpikeInterface is centered around
    extracellular recordings, and this function is designed to convert the data to microvolts (µV).
2. ``scale_to_physical_units``: A general function for any physical unit conversion. This will allow you to extract the data in any
    physical unit, not just microvolts. This is useful for other types of recordings, such as force measurements in Newtons but should be
    handled with care.

For most users working with extracellular recordings, ``scale_to_uV`` is the recommended choice if they want to work in physical units:

.. code-block:: python

    from spikeinterface.extractors import read_intan
    from spikeinterface.preprocessing import scale_to_uV

    # Load recording (data is in ADC units)
    recording = read_intan("path/to/file.rhs")

    # Convert to microvolts
    recording_uv = scale_to_uV(recording)

For recordings with non-standard units (e.g., force measurements in Newtons), use ``scale_to_physical_units``:

.. code-block:: python

    from spikeinterface.preprocessing import scale_to_physical_units

    # Convert to physical units (whatever they may be)
    recording_physical = scale_to_physical_units(recording)

Both preprocessors automatically:

1. Detect the appropriate gain and offset from the recording properties
2. Apply the conversion to all channels
3. Update the recording properties to reflect that data is now in physical units

Setting Custom Physical Units
-----------------------------

While most extractors automatically set the appropriate ``gain_to_uV`` and ``offset_to_uV`` values,
there might be cases where you want to set custom physical units. In these cases, you can set
the following properties:

* ``physical_unit``: The target physical unit (e.g., 'uV', 'mV', 'N')
* ``gain_to_unit``: The gain to convert from raw values to the target unit
* ``offset_to_unit``: The offset to convert from raw values to the target unit

You need to set these properties for every channel, which allows for the case when there are different gains and offsets on different channels. Here's an example:

.. code-block:: python

    # Set custom physical units
    num_channels = recording.get_num_channels()
    values = ["volts"] * num_channels
    recording.set_property(key='physical_unit', values=values)

    gain_values = [0.001] * num_channels  # Convert from ADC to volts
    recording.set_property(key='gain_to_unit', values=gain_values)  # Convert to volts

    offset_values = [0] * num_channels  # No offset
    recording.set_property(key='offset_to_unit', values=offset_values)  # No offset

    # Apply the conversion using scale_to_physical_units
    recording_physical = scale_to_physical_units(recording)

This approach gives you full control over the unit conversion process while maintaining
compatibility with SpikeInterface's preprocessing pipeline.
