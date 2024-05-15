Export MATLAB Data to Binary & Load in SpikeInterface
========================================================

In this tutorial, we will walk through the process of exporting data from MATLAB in a binary format and subsequently loading it using SpikeInterface in Python.

Exporting Data from MATLAB
--------------------------

Begin by ensuring your data structure is correct. Organize your data matrix so that the first dimension corresponds to samples/time and the second to channels.
Here, we present a MATLAB code that creates a random dataset and writes it to a binary file as an illustration.

.. code-block:: matlab

   % Define the size of your data
   numSamples = 1000;
   numChannels = 384;

   % Generate random data as an example
   data = rand(numSamples, numChannels);

   % Write the data to a binary file
   fileID = fopen('your_data_as_a_binary.bin', 'wb');
   fwrite(fileID, data, 'double');
   fclose(fileID);

.. note::

   In your own script, replace the random data generation with your actual dataset.

Loading Data in SpikeInterface
------------------------------

After executing the above MATLAB code, a binary file named :code:`your_data_as_a_binary.bin` will be created in your MATLAB directory. To load this file in Python, you'll need its full path.

Use the following Python script to load the binary data into SpikeInterface:

.. code-block:: python

   import spikeinterface as si
   from pathlib import Path

   # Define file path
   # For Linux or macOS:
   file_path = Path("/The/Path/To/Your/Data/your_data_as_a_binary.bin")
   # For Windows:
   # file_path = Path(r"c:\path\to\your\data\your_data_as_a_binary.bin")

   # Confirm file existence
   assert file_path.is_file(), f"Error: {file_path} is not a valid file. Please check the path."

   # Define recording parameters
   sampling_frequency = 30_000.0  # Adjust according to your MATLAB dataset
   num_channels = 384  # Adjust according to your MATLAB dataset
   dtype = "float64"  # MATLAB's double corresponds to Python's float64

   # Load data using SpikeInterface
   recording = si.read_binary(file_paths=file_path, sampling_frequency=sampling_frequency,
                              num_channels=num_channels, dtype=dtype)

   # Confirm that the data was loaded correctly by comparing the data shapes and see they match the MATLAB data
   print(recording.get_num_frames(), recording.get_num_channels())

Follow the steps above to seamlessly import your MATLAB data into SpikeInterface. Once loaded, you can harness the full power of SpikeInterface for data processing, including filtering, spike sorting, and more.

Common Pitfalls & Tips
----------------------

1. **Data Shape**: Make sure your MATLAB data matrix's first dimension is samples/time and the second is channels. If your time is in the second dimension, use :code:`time_axis=1` in :code:`si.read_binary()`.
2. **File Path**: Always double-check the Python file path.
3. **Data Type Consistency**: Ensure data types between MATLAB and Python are consistent. MATLAB's `double` is equivalent to Numpy's `float64`.
4. **Sampling Frequency**: Set the appropriate sampling frequency in Hz for SpikeInterface.
5. **Transition to Python**: Moving from MATLAB to Python can be challenging. For newcomers to Python, consider reviewing numpy's `Numpy for MATLAB Users <https://numpy.org/doc/stable/user/numpy-for-matlab-users.html>`_ guide.

Using gains and offsets for integer data
----------------------------------------

Raw data formats often store data as integer values for memory efficiency. To give these integers meaningful physical units, you can apply a gain and an offset.
In SpikeInterface, you can use the :code:`gain_to_uV` and :code:`offset_to_uV` parameters, since traces are handled in microvolts (uV). Both parameters can be integrated into the :code:`read_binary` function.
If your data in MATLAB is stored as :code:`int16`, and you know the gain and offset, you can use the following code to load the data:

.. code-block:: python

   sampling_frequency = 30_000.0  # Adjust according to your MATLAB dataset
   num_channels = 384  # Adjust according to your MATLAB dataset
   dtype_int = 'int16'  # Adjust according to your MATLAB dataset
   gain_to_uV = 0.195  # Adjust according to your MATLAB dataset
   offset_to_uV = 0   # Adjust according to your MATLAB dataset

   recording = si.read_binary(file_paths=file_path, sampling_frequency=sampling_frequency,
                              num_channels=num_channels, dtype=dtype_int,
                              gain_to_uV=gain_to_uV, offset_to_uV=offset_to_uV)

   recording.get_traces()  # Return traces in original units [type: int]
   recording.get_traces(return_scaled=True)  # Return traces in micro volts (uV) [type: float]


This will equip your recording object with capabilities to convert the data to float values in uV using the :code:`get_traces()` method with the :code:`return_scaled` parameter set to :code:`True`.

.. note::

   The gain and offset parameters are usually format dependent and you will need to find out the correct values for your data format. You can load your data without gain and offset but then the traces will be in integer values and not in uV.
