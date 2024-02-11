Install
=======

As for now, there are two ways to install the package.

PyPI
----

Using PyPI, it suffices to run :code:`pip install uotod`. Just rerun this command to update the package to its newest version.


Build From Source
-----------------

You can also download it directly from the GitHub repository, then build and install it.

.. code-block:: bash

    git clone --recursive https://github.com/hdeplaen/uotod
    cd uotod
    python3 -m pip install -r requirements.txt
    python3 -m setup build
    python3 -m pip install
