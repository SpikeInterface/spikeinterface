Working with physical units in SpikeInterface recordings
========================================================

In neurophysiology recordings, data is often stored in raw ADC (Analog-to-Digital Converter) units
but needs to be analyzed in physical units. For extracellular recordings, this is typically microvolts (µV),
but some recording devices may use different physical units. SpikeInterface provides tools to handle both
situations.

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
   extracellular recordings, and this function is designed to convert the data to microvolts (µV). Many plotting
   and analyzing functions in SpikeInterface expect data in microvolts, so this is the recommended approach for most users.
2. ``scale_to_physical_units``: A general function for any physical unit conversion. This will allow you to extract the data in any
   physical unit, not just microvolts. This is useful for other types of recordings, such as force measurements in Newtons but should be
   handled with care.

For most users working with extracellular recordings, ``scale_to_uV`` is the recommended choice:

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
---------------------------

While most extractors automatically set the appropriate ``gain_to_uV`` and ``offset_to_uV`` values,
there might be cases where you want to set custom physical units. In these cases, you can set
the following properties:

* ``physical_unit``: The target physical unit (e.g., 'uV', 'mV', 'N')
* ``gain_to_unit``: The gain to convert from raw values to the target unit
* ``offset_to_unit``: The offset to convert from raw values to the target unit

Here's an example of how to set custom units:

.. code-block:: python

    # Set custom physical units
    num_channels = recording.get_num_channels()
    values = ["volts"] * num_channels
    recording.set_property(key='physical_unit', value=values)

    values = [0.001] * num_channels  # Convert from ADC to volts
    recording.set_property(key='gain_to_unit', values=values)  # Convert to volts

    values = [0] * num_channels  # No offset
    recording.set_property(key='offset_to_unit', values=values)  # No offset

    # Apply the conversion using scale_to_physical_units
    recording_physical = scale_to_physical_units(recording)

This approach gives you full control over the unit conversion process while maintaining
compatibility with SpikeInterface's preprocessing pipeline.
