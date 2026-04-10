# SAFIRE
--------


## Documentation
----------------

The user documentation can be built using sphinx as follows.

First, install sphinx. 
This is done automatically if afqmctools is installed as described [here](utils/README.md#installation) using the `DOCS` opptional dependency set.
Alternatively, sphinx can be installed via pip using
```
pip install [-U] sphinx sphinxcontrib-bibtex sphinx-rtd-theme myst_nb sphinx-autodoc-typehints numpydoc
```
where `-U` should be used if installing outside of a virtual environment.
Using a virtual enviroment is generally recommended,
especially on rusty or an SCC-managed workstation.

Next, build the docs by running the following starting from the root of this repository:
```bash
$ cd docs
$ make html
```
some error messages and warnings may occur, but this should generate an html document in `_build/html`.

Finally, to view the documentation locally, navigate in a browser to the documentation using:

```
file:///path/to/SAFIRE/docs/_build/html/index.html
```

## Installation
---------------

This project uses submodules which must be initialized.
This can be achievied using the following commands:

```bash
$ git submodule init
$ git submodule update
```

Then, make an out-of-source build directory:

```bash
$ mkdir build
$ cd build
```

Next, follow one of the sections below depending on whether you want a CPU-only build,
or a GPU-accelerated build.


Also, see the afqmctools Python package [Readme](utils/README.md#installation) to install the afqmctools Pyhton package.


### Dependencies
----------------

For a CPU-only build:

- a compiler that supports c++17
- cmake
- MPI
- HDF5
- Boost versions 1.61 or more recent
- BLAS / LAPACK

For Tests
- CTest

For a GPU-build, the following are also required:


- CUDA 11 or 12
- A GPU with compute compatability >=8 (see [#125])

### CPU-only build at CCQ
-------------------------

For large systems, building and running with GPU acceleration is highly recommended.
However, for smaller systems, CPU builds are useful.

If you are rusty or sitting at an SCC-managed workstation, a suitable build script is

```bash
module purge
module load slurm
module load cmake
module load gcc
module load openmpi
module load hdf5
module load boost
module load intel-oneapi-mkl
module load python-mpi/3.11


cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DHAVE_MKL=ON 

make -j 10

```

### GPU accelerated build at CCQ
--------------------------------

See [#125] for the latest updates on GPU support.

#### CUDA 12
This version is the recommended version to use with the latest version of Jax/AutoHF.  
For CUDA 12, you can use the following on rusty

```bash
module purge
# CPU
module load slurm
module load cmake
module load gcc/12
module load openmpi
module load hdf5
module load boost
module load intel-oneapi-mkl
module load python-mpi/3.11
# GPU
module load cuda/12

cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DHAVE_MKL=ON \
        -DENABLE_CUDA=ON 


make -j 10
```

#### CUDA 11

If you are on rusty or using a rusty connected desktop a suitable build script is

```bash
module purge
# CUDA 11 is gone from modules/2.3+
module swap modules modules/2.2-20230808

# CPU
module load slurm
module load cmake
module load gcc/11
module load openmpi
module load hdf5
module load boost
module load intel-oneapi-mkl

# GPU
module load cuda/11

# build
module load python-mpi/3.11
module load cmake

cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DHAVE_MKL=ON \
        -DENABLE_CUDA=ON 


make -j 10
```

## License
----------

This project is licensed under the [Apache License, Version 2.0](LICENSE).

Portions of this software are derived from the QMCPACK
project, which was originally distributed under the [University of Illinois/NCSA Open Source License](LICENSES/NCSA.txt).
The full text of that license can be found in [LICENSES/NCSA.txt](LICENSES/NCSA.txt).
