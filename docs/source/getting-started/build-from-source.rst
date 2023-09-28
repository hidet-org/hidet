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

Build shared libraries
~~~~~~~~~~~~~~~~~~~~~~

The runtime library is written in C++ and compiled into a shared library. To build the shared library, you need to have
a C++ compiler installed (as well as build tools like ``cmake``, and ``make``). The following command will build the
shared library:

.. code-block:: console

  $ cd hidet
  $ mkdir build
  $ cd build
  $ cp ../config.cmake .  # copy the cmake config to build directory
  $ cmake ..
  $ make -j4

After building, you could find two libraries ``libhidet.so`` and ``libhidet_runtime.so`` under ``build/lib`` directory.

Install the Hidet Python package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next we will install the Python package of Hidet in the develop mode via pip:

.. code-block:: console

  $ cd .. # return to the root directory of Hidet
  $ pip install -e .

Validation
~~~~~~~~~~

To make sure we have successfully installed hidet, run the following command in a new shell:

.. code-block:: console

  $ python -c "import hidet"

If no error reports, then hidet has been successfully installed on your computer.
