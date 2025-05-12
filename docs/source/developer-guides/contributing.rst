Contributing
============

To contribute to this project, please fork the hidet repository and create a pull request.
Before submitting a pull request, please make sure that your code is properly formatted and that it passes the tests.

**Install editable dev version of hidet**

.. code-block:: bash

    $ git clone hidet-org/hidet
    $ cd hidet
    $ pip install -e .[dev]

**Format & Lint** Run the following scripts to format and lint the code:

.. code-block:: bash

    $ bash scripts/lint/format.sh
    $ bash scripts/lint/lint.sh

**Tests** To run the tests, run the following script:

.. code-block:: bash

    $ # use --clear-cache to clear the operator cache if you changed the following sub-modules
    $ #  - hidet.ir
    $ #  - hidet.backend
    $ pytest tests

