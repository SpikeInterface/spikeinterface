Exporting MATLAB Data to Binary & Loading in SpikeInterface
===========================================================

In this tutorial, we'll go through the process of exporting your data from MATLAB in a binary format and then loading it using SpikeInterface in Python. Let's break down the steps.

Exporting Data from MATLAB
--------------------------

First, ensure your data is structured correctly. The data matrix should be organized such that the first dimension corresponds to samples/time and the second dimension to channels.
In the following MATLAB code, we generate random data as an example and then write it to a binary file.

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

   In a real-world scenario, replace the random data generation with your actual data.

Loading Data in SpikeInterface
-----------------------------

This should produce a binary file called `your_data_as_a_binary.bin` in your current MATLAB directory.
You will need the complete path (i.e. its location on your computer) to load it in Python.

Once you have your data in a binary format, you can seamlessly load it into SpikeInterface using the following script:

.. code-block:: python

   import spikeinterface as si
   from pathlib import Path

   # In linux or mac
   file_path = Path("/The/Path/To/Your/Data/your_data_as_a_binary.bin")
   # or for Windows
   # file_path = Path(r"c:\path\to\your\data\your_data_as_a_binary.bin")

   # Ensure the file exists
   assert file_path.is_file(), f"Your path {file_path} is not a file, you probably have a typo or got the wrong path."

   # Specify the parameters of your recording
   sampling_frequency = 30_000.0  # in Hz, adjust as per your MATLAB dataset
   num_channels = 384  # adjust as per your MATLAB dataset
   dtype = "float64"  # equivalent of MATLAB double

   # Load the data using SpikeInterface
   recording = si.read_binary(file_path, sampling_frequency=sampling_frequency,
                                        num_channels=num_channels, dtype=dtype)

   # Verify the shape of your data
   assert recording.get_traces().shape == (num_samples, num_channels)

This should be enough to get you started with loading your MATLAB data into SpikeInterface. You can use all the Spikeinterface machinery to process your data, including filtering, spike sorting, and more.

Common Pitfalls & Tips
----------------------

1. **Data Shape**: Always ensure that your MATLAB data matrix's first dimension corresponds to samples/time and the second to channels. If the time happens to be in the second dimension, you can use `time_axis=1` as an argument in `si.read_binary()` to account for this.
2. **File Path**: Double-check the file path in Python to ensure you are pointing to the right directory.
3. **Data Type**: When moving data between MATLAB and Python, it's crucial to keep the data type consistent. In our example, we used `double` in MATLAB, which corresponds to `float64` in Python.
4. **Sampling Frequency**: Ensure you set the correct sampling frequency in Hz when loading data into SpikeInterface.
5. **Working on Python**: Matlab to python can feel like a big jump. If you are new to Python, we recommend checking out numpy's [Python for MATLAB Users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html) guide.


Using gains and offsets for integer data
----------------------------------------

A common technique used in raw formats is to store data as integer values, which provides a memory-efficient representation (i.e. lower ram) and use a gain and offset to convert it to float values that represent meaningful physical units.
In SpikeInterface this is done using the `gain_to_uV` and `offset_to_uV` parameters as the we handle traces in microvolts. Both values can be passed to `read_binary` when loading the data:

.. code-block:: python

   sampling_frequency = 30_000.0  # in Hz, adjust as per your MATLAB dataset
   num_channels = 384  # adjust as per your MATLAB dataset
   dtype_int = 'int16'  # adjust as per your MATLAB dataset
   gain_to_uV = 0.195  # adjust as per your MATLAB dataset
   offset_to_uV = 0   # adjust as per your MATLAB dataset

   recording = si.read_binary(file_path, sampling_frequency=sampling_frequency,
                              num_channels=num_channels, dtype=dtype_int,
                              gain_to_uV=gain_to_uV, offset_to_uV=offset_to_uV)

   recording.get_traces(start)


This will equip your recording object with capabilities to convert the data to float values in uV using the `get_traces()` method with the `return_scaled` parameter set to True.
