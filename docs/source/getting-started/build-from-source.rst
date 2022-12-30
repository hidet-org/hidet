Build from source
-------------------
.. _Build-from-source:

If you want to contribute to Hidet, or you encountered any problem installing hidet via pip, it is better to install
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

Update environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To allow Python interpreter to find hidet package under ``python`` directory of the repository, we should append the
directory to ``PYTHONPATH`` variable. To allow the system find the shared libraries we built in the previous step,
we should append ``build/lib`` directory to ``LD_LIBRARY_PATH`` variable.

.. code-block:: console

  $ export HIDET_HOME=<The Path to Hidet Repo>
  $ export PYTHONPATH=$PYTHONPATH:$HIDET_HOME/python
  $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HIDET_HOME/build/lib

To avoid repeating above commands, it is recommended to put above commands to your shell's initialization script
(e.g., ``~/.bashrc`` for Bash and ``~/.zshrc`` for Zsh).

Validation
~~~~~~~~~~

To make sure we have successfully installed hidet, run the following command in a new shell:

.. code-block:: console

  $ python -c "import hidet"

If no error reports, then hidet has been successfully installed on your computer.
