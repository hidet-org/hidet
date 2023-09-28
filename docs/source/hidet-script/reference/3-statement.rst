Statements
==========

Control flow statements
-----------------------

Hidet script supports the following control flow statements:

- ``if`` statement
- ``for`` statement
  - ``for-mapping`` statement
  - ``while`` statement

If statement
------------

The ``if`` statement has the same semantics as the ``if`` statement in c/c++ and python.

For statement
-------------

Hidet script supports the following kinds of for statements:

.. code-block::

    from hidet.lang import grid, printf

    for i in range(10):
        printf("%d\n", i)

    for i, j in grid(10, 10):
        printf("%d %d\n", i, j)

    for indices in grid(10, 10, bind_tuple=True):
        printf("%d %d\n", indices[0], indices[1])

    # explicitly set the attributes of the loop variables
    # the attribute can be one of
    # - 'p': parallelize this loop axis
    # - 'u': unroll this loop axis
    # a number can be appended to the attribute to specify how many threads to use or the unroll factor
    # like 'p2u3' means parallelize loop axis 'i' with 2 threads and unroll loop axis 'j' with factor 3.
    for i, j in grid(10, 10, attrs='pu'):
        printf("%d %d\n", i, j)


For mapping statement
---------------------

Task mapping
~~~~~~~~~~~~

Please refer to the `Hidet paper <https://dl.acm.org/doi/10.1145/3575693.3575702>`_ for the definition of task mapping

.. todo::

  add a brif introduction here to make it self-contained

Iterate the task mapping
~~~~~~~~~~~~~~~~~~~~~~~~

The task mappings are defined in the ``hidet.lang.mapping`` module. To use the task mappings, we can import the module
and use the task mappings like the following:

.. code-block::

    from hidet.lang import printf, grid
    from hidet.lang.mapping import spatial, repeat

    # iterate the spatial mapping
    for w in grid(10, attrs='p'):
        for i, j in spatial(2, 5).on(w):
            printf("%d %d\n", i, j)

        for i, j in spatial(2, 5).repeat(3, 4).on(w):
            # task mapping shape: (6, 20)
            # num workers: 10
            printf("%d %d\n", i, j)

While statement
---------------

Hidet also supports the ``while`` statement, and it has the same semantics as python and c/c++.
