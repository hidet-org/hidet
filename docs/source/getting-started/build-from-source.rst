Build from source
-------------------
.. _Build-from-source:

If you want to contribute to Hidet, or you encountered any problem directly installing hidet via pip, it is better to install
hidet from source.

Clone the code
~~~~~~~~~~~~~~

First clone the repository to local:

.. code-block:: console

  $ git clone https://github.com/hidet-org/hidet
  $ cd hidet    # enter the hidet directory

Install the Hidet Python package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next we will install the Python package of Hidet via pip. The following command will install the package in the develop
mode, in which the modification of the source code will be immediately reflected in the installed package. If you want to
install the package in the normal mode, use 'pip install .' instead.

.. code-block:: console

  $ pip install -e .

Validation
~~~~~~~~~~

To make sure we have successfully installed hidet, run the following command in a new shell:

.. code-block:: console

  $ python -c "import hidet"

If no error reports, then hidet has been successfully installed on your computer.
