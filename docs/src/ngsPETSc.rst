ngsPETSc
------------------

ngsPETSc is a PETSc interface for NGSolve.

Installation
-----------------
To install ngsPETSc you need to clone the GitHub repository, and then you can install it using pip.
::
    git clone https://github.com/UZerbinati/ngsPETSc.git
    cd ngsPETSc
    pip install .
You can also build PETSc, SLEPc and NGSolve from source and then install ngsPETSc.
First we install all the needed package using apt and pip or an equivalent package manager.
::
    apt-get update
    apt-get -y install git build-essential cmake python3 python3-distutils python3-tk libpython3-dev libxmu-dev tk-dev tcl-dev g++ libglu1-mesa-dev liblapacke-dev libblas-dev liblapack-dev
    pip install numpy cython pytest pytest-mpi

We now install PETSc from scratch in a home installation folder, with OpenMPI, HYPRE, Metis, MUMPS, SuprLU, Scalapack and eigen.
::
    git clone https://gitlab.com/petsc/petsc.git
    cd petsc
    python configure --download-chaco \
    --download-cmake \
    --download-eigen \
    --download-openmpi \
    --download-hypre \
    --download-metis \
    --download-ml \
    --download-mumps \
    --download-scalapack \
    --download-superlu_dist \
    --with-c2html=0 \
    --with-cxx-dialect=C++11 \
    --with-debugging=0 \
    --download-fblaslapack=1 \
    --with-fortran-bindings=0 \
    --with-shared-libraries=1 \
    --with-petsc4py=1 \
To build PETSc you need to run the Makefile as suggested at the end of the configuration script.
We now need to set in the `.bashrc` (for MacOS user `.bash_profile`) file the `PETSC_DIR`, `PETSC_ARCH` system variable as they appear when we finish build PETSc.
You also need to add to your `PYTHONPATH` the `PYTHONPATH` appear when we finished building PETSc.
We also suggest adding the following line to you `.bashrc` the following lines:
::
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR/$PETSC_ARCH/lib
    export PATH=$PATH:$PETSC_DIR/$PETSC_ARCH/bin 

We now install SLEPc from source once again in the home installation folder
::
    git clone https://gitlab.com/slepc/slepc.git
    cd slepc
    python configure --download-blopex --with-slepc4py=1
To build SLEPc you need to run the Makefile as suggested at the end of the configuration script.
We now need to set in the `.bashrc` (for MacOS user `.bash_profile`) file the `SLEPC_DIR` system variable as they appear when we finish build PETSc.
You also need to add to your `PYTHONPATH` the `PYTHONPATH` appear when we finished building SLEPc.
We now build mpi4py form source (once again in our home installation folder) in order to have a mpi4py installation that uses PETSc's local MPI installation.
::
    git clone https://github.com/mpi4py/mpi4py.git
    cd mpi4py
    pip install .

Now we are left building NGSolve from sources, once again in the home installation directory.
::
    export BASEDIR=$PWD/ngsuite
    mkdir -p $BASEDIR
    cd $BASEDIR
    git clone https://github.com/NGSolve/ngsolve.git ngsolve-src
    cd $BASEDIR/ngsolve-src
    git submodule update --init --recursive
    mkdir $BASEDIR/ngsolve-build
    mkdir $BASEDIR/ngsolve-install
    cd $BASEDIR/ngsolve-build
    cmake -DCMAKE_INSTALL_PREFIX=${BASEDIR}/ngsolve-install ${BASEDIR}/ngsolve-src -DUSE_MPI=ON
    make
    make install

We suggest you adding following lines to your `.bashrc`:
::
    export NETGENDIR="${BASEDIR}/ngsolve-install/bin"
    export PATH=$NETGENDIR:$PATH
    export PYTHONPATH=$NETGENDIR/../`python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib(1,0,''))"`


Authors
----------

Umberto Zerbinati

License
---------------

The package is released under the `MIT
License <https://opensource.org/licenses/MIT>`__.

API
----

.. automodule:: ngsPETSc
   :members:
