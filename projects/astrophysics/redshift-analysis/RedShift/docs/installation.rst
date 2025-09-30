Installation
============

Requirements
------------

The QFD CMB Module requires Python 3.8 or later and the following dependencies:

* numpy >= 1.19.0
* scipy >= 1.6.0
* matplotlib >= 3.3.0

Optional dependencies for development:

* pytest >= 6.0.0 (for running tests)
* sphinx >= 4.0.0 (for building documentation)
* jupyter >= 1.0.0 (for running tutorials)

Installing from PyPI
---------------------

The easiest way to install the QFD CMB Module is using pip:

.. code-block:: bash

   pip install qfd-cmb

This will automatically install all required dependencies.

Installing from Source
----------------------

To install the latest development version from source:

.. code-block:: bash

   git clone https://github.com/qfd-project/qfd-cmb.git
   cd qfd-cmb
   pip install -e .

For development installation with all optional dependencies:

.. code-block:: bash

   git clone https://github.com/qfd-project/qfd-cmb.git
   cd qfd-cmb
   pip install -e ".[dev]"

Verifying Installation
----------------------

To verify that the installation was successful, run:

.. code-block:: python

   import qfd_cmb
   print(qfd_cmb.__version__)

You can also run the test suite:

.. code-block:: bash

   pytest tests/

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'qfd_cmb'**

This usually means the package wasn't installed correctly. Try reinstalling:

.. code-block:: bash

   pip uninstall qfd-cmb
   pip install qfd-cmb

**Numerical precision warnings**

If you see warnings about numerical precision, this is usually harmless but can be 
suppressed by setting appropriate numpy error handling:

.. code-block:: python

   import numpy as np
   np.seterr(divide='ignore', invalid='ignore')

**Matplotlib backend issues**

If you encounter issues with plotting, try setting a non-interactive backend:

.. code-block:: python

   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt

Platform-Specific Notes
~~~~~~~~~~~~~~~~~~~~~~~~

**Windows**

On Windows, you may need to install Microsoft Visual C++ Build Tools if you encounter 
compilation errors during installation.

**macOS**

On macOS, ensure you have Xcode command line tools installed:

.. code-block:: bash

   xcode-select --install

**Linux**

Most Linux distributions should work out of the box. If you encounter issues with 
scientific Python packages, consider using conda instead of pip.