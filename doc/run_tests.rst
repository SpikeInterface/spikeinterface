Running tests
=============

Running pytest requires the test extra:

.. code-block:: bash

  pip install spikeinterface[test]

Note that if using Z shell (:code:`zsh` - the default shell on mac), you will need to use quotes (:code:`pip install "spikeinterface[test]"`).

.. tip::

  Make a folder name `test_results` on the root of the repository, and run the tests from here :code:`poetry run pytest ..`
