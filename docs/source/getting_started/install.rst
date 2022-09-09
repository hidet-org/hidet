Installation
============

Install from pip
----------------

.. note::
  :class: margin

  Currently, hidet is not open-sourced yet. Please ask Yaoyao get access to the GitHub repository.

  In the future, we will publish hidet to The `Python Package Index (PyPI) <https://pypi.org/>`_.

Run the following command to install ``hidet`` package directly using python pip:

.. code-block:: console

  $ pip install git+ssh://git@github.com/yaoyaoding/hidet.git


Install from source
-------------------

If you are a developer of Hidet, it is better to install hidet from source.

Clone the code
~~~~~~~~~~~~~~

First clone the repository to local:

.. code-block:: console

  $ git clone https://github.com/yaoyaoding/hidet

Build shared libraries
~~~~~~~~~~~~~~~~~~~~~~

Create the directory to build the shared libraries:

.. code-block:: console

  $ cd hidet
  $ mkdir build
  $ cd build
  $ cp ../config.cmake .  # copy the cmake config to build directory

You could customize the build by update the ``config.cmake`` file, according the instructions in the default config file. Usually,
just leave it as default.

Build hidet shared libraries (under ``build`` directory):

.. code-block:: console

  $ cmake ..
  $ make -j4

After building, you could find two libraries ``libhidet.so`` and ``libhidet_runtime.so`` under ``build/lib`` directory.

Update environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To allow Python interpreter to find hidet package under ``python`` directory of the repository, we should append the directory to ``PYTHONPATH`` variable.
To allow the system find the shared libraries we built in the previous step, we should append ``build/lib`` directory to ``LD_LIBRARY_PATH`` variable.

.. code-block:: console

  $ export HIDET_HOME=<The Path to Hidet Repo>
  $ export PYTHONPATH=$PYTHONPATH:$HIDET_HOME/python
  $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HIDET_HOME/build/lib

To avoid repeating above commands, it is recommended to put above commands to your shell's initialization script (e.g., ``~/.bashrc`` for Bash and ``~/.zshrc`` for Zsh).

Validation
~~~~~~~~~~

To make sure we have successfully installed hidet, run the following command in a new shell:

.. code-block:: console

  $ python -c "import hidet"

If no error reports, then hidet has been successfully installed on your computer.


