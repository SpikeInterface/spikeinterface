.. _unsigned_to_signed:

Handle unsigned data types
==========================

As of version 0.103.0 SpikeInterface has changed one of its defaults for interacting with
:code:`Recording` objects. We no longer autocast unsigned dtypes to signed implicitly. This
means that some users of SpikeInterface will need to add one additional line of code to their scripts
to explicitly handle this conversion.


Why this matters?
-----------------

For those that want a deeper understanding of dtypes `NumPy provides a great explanation <https://numpy.org/doc/stable/reference/arrays.dtypes.html>`_.
For our purposes it is important to know that many pieces of recording equipment opt to store their electrophysiological data as unsigned integers
(e.g., Intan, Maxwell Biosystems, 3Brain Biocam).
Similarly to signed integers, in order to convert to real units these file formats only need to store a :code:`gain`
and an :code:`offset`. Our :code:`RecordingExtractor`'s maintain the dtype that the file format utilizes, which means that some of our
:code:`RecordingExtractor`'s will have unsigned dtypes.

The problem with using unsigned dtypes is that many types of functions (including the ones we use from :code:`SciPy`) perform poorly with unsigned integers.
This is made worse by the fact that these failures are silent (i.e. no error is triggered but the operation leads to nonsensical data). So the
solution required is to convernt unsigned integers into signed integers. Previously we did this under the hood, automatically for users that had
a :code:`Recording` object with an unsigned dtype.

We decided, however, that implicitly performing this action was not the best course of action, since:

1) *explicit* is always better than *implicit*
2) some functions would *magically* change the dtype of the :code:`Recording` object, which can cause confusion

So from version 0.103.0, users will now explicitly have to perform this transformation of their data. This will help users better understand how they are
processing their data during an analysis pipeline as well as better understand the provenance of their pipeline.


Using :code:`unsigned_to_signed`
--------------------------------

For users that receive an error because their :code:`Recording` is unsigned, their is one additional step that must be done:

.. code:: python

    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as spre

    # Intan is an example of unsigned data
    recording = se.read_intan('path/to/my/file.rhd', stream_id='0')
    # to get a signed version of our Recording we use the following function
    recording_signed = spre.unsigned_to_signed(recording)
    # we can now apply any preprocessing functions like normal, e.g.
    recording_filtered = spre.bandpass_filter(recording_signed)


Now with the signed dtype of the :code:`Recording` one can use a SpikeInterface pipeline as usual.


If you are curious if your :code:`Recording` is unsigned you can simply check the repr or use :code:`get_dtype()`

.. code:: python

    # the repr automatically displays the dtype
    print(recording)
    # use method on the Recording object
    print(recording.get_dtype())

In either case, if the dtype displayed has a :code:`u` at the beginning (e.g. :code:`uint16`) then your recording is
unsigned. If it doesn't have the :code:`u` (e.g. :code:`int16`) then it is signed and would not need this preprocessing step.


Bit depth
---------

One final important piece of information for some users is the concept of bit depth, which is the number of bits used to
sample the data. The :code:`bit_depth` argument that can be fed into the :code:`unsigned_to_signed` function.
This should be used in cases where the ADC bit depth does not match the bit depth of the data type (e.g., if the data is
stored as :code:`uint16` but the ADC is 12 bits).
Let's make a concrete example: the Biocam acquisition system from 3Brain uses a 12-bit ADC and stores the data as
:code:`uint16`. This means that the data is stored in a 16-bit unsigned integer format, but the actual data
only covers a 12-bit range. Therefore, that the "zero" of the data is not at 0, nor at half of the :code:`uint16` range (i.e. 2^15),
but rather at 2048 (i.e., 2^12).
In this case, setting the :code:`bit_depth` argument to 12 will allow the :code:`unsigned_to_signed` function to
correctly convert the unsigned data to signed data and offset the data to be centered around 0, by subtracting 2048
while converting the data from unsigned to signed.

.. code:: python

    recording_unsigned = se.read_biocam('path/to/my/file.brw')
    # we can now convert to signed with the correct bit depth
    recording_signed = spre.unsigned_to_signed(recording_unsigned, bit_depth=12)


Additional Notes
----------------

1) Some sorters make use of SpikeInterface preprocessing either
   within their wrappers or within their own code base. So remember to use the "signed" version of
   your recording for the rest of your pipeline.

2) Using :code:`unsigned_to_signed` in versions less than 0.103.0 does not hurt your scripts. This
   option was available previously along with the implicit option. Adding this into scripts with old
   versions of SpikeInterface will still work and will "future-proof" your scripts for when you
   update to a version greater than or equal to 0.103.0.

3) For additional information on units and scaling in SpikeInterface see :ref:`physical_units`.
