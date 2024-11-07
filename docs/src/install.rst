Installation
-----------------
To install ngsPETSc you need to clone the GitHub repository, and then you can install it using pip.
::
    git clone https://github.com/UZerbinati/ngsPETSc.git
    cd ngsPETSc
    pip install .

Alternatively, you can also build PETSc, SLEPc and NGSolve from source and then install ngsPETSc.
First we install all the needed packages using apt and pip or an equivalent package manager.
::
    apt-get update
    apt-get -y install git build-essential cmake python3 python3-distutils python3-tk libpython3-dev libxmu-dev tk-dev tcl-dev g++ libglu1-mesa-dev liblapacke-dev libblas-dev liblapack-dev
    pip install numpy cython pytest pytest-mpi netgen-occt

We now install PETSc from scratch in a suitable folder, with OpenMPI, HYPRE, Metis, MUMPS, SuprLU, Scalapack and eigen.
::
    git clone https://gitlab.com/petsc/petsc.git
    cd petsc
    python configure --download-cmake \
    --download-openmpi \
    --download-hypre \
    --download-metis \
    --download-parmetis \
    --download-ml \
    --download-mumps \
    --download-scalapack \
    --download-superlu_dist \
    --download-fblaslapack=1 \
    --with-c2html=0 \
    --with-debugging=0 \
    --with-fortran-bindings=0 \
    --with-shared-libraries=1 \
    --with-petsc4py=1

To build PETSc you need to run the Makefile as suggested at the end of the configuration script.
We now need to set in the ``.bashrc`` (on OSX in ``.bash_profile``) file the ``PETSC_DIR``, ``PETSC_ARCH`` system variables as they appear when we finish build PETSc.
You also need to add to your ``PYTHONPATH`` the ``PYTHONPATH`` that appears when we finished building PETSc.
We also suggest adding the following line to your ``.bashrc`` the following lines:
::
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR/$PETSC_ARCH/lib
    export PATH=$PATH:$PETSC_DIR/$PETSC_ARCH/bin 

We now install SLEPc from source.
::
    git clone https://gitlab.com/slepc/slepc.git
    cd slepc
    python configure --download-blopex --with-slepc4py=1
To build SLEPc you need to run the Makefile as suggested at the end of the configuration script.
Again set in the ``.bashrc`` (for MacOS user ``.bash_profile``) file the ``SLEPC_DIR`` system variable as it appears when we finish build SLEPc.
You also need to add to your ``PYTHONPATH`` the ``PYTHONPATH`` that appears when we finished building SLEPc.
We now build mpi4py from source in order to have an mpi4py installation that uses PETSc's local MPI installation.
::
    git clone https://github.com/mpi4py/mpi4py.git
    cd mpi4py
    pip install .

Now we build NGSolve from source.
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
    cmake -DCMAKE_INSTALL_PREFIX=${BASEDIR}/ngsolve-install ${BASEDIR}/ngsolve-src -DUSE_MPI=ON -DUSE_OCC=ON
    make
    make install

You should add to your ``.bashrc`` the ``BASEDIR`` system variable:
::
    echo "export $BASEDIR=${BASEDIR}" >> ~/.bashrc  

We suggest you add the following lines to your ``.bashrc``:
::
    export NETGENDIR="${BASEDIR}/ngsolve-install/bin"
    export PATH=$NETGENDIR:$PATH
    export PYTHONPATH=$PYTHONPATH:$NETGENDIR/../`python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib(1,0,''))"`

We are now finally ready to install ngsPETSc:
:: 
    git clone https://github.com/UZerbinati/ngsPETSc.git
    cd ngsPETSc
    NGSPETSC_NO_INSTALL_REQUIRED=ON pip install .

Contributing
-------------

🎉**Thanks for taking the time to contribute!** 🎉

To get an overview of the project, check out the [README](README.md).

The `issue tracker <https://github.com/NGSolve/ngsPETSc/issues>`__.
is the preferred channel for bug reports.

A bug is a demonstrable problem that is caused by the code in the repository.
Bug reports are extremely helpful - thank you!

Guidelines for bug reports:

1. **Check if the issue has been fixed**: try to reproduce it using the latest `main` or development branch in the repository.

2. **Use the GitHub issue search**: check if the issue has already been reported.

3. **Isolate the problem**: Create a minimal example showing the problem.

4. **Open an issue**: Using the `issue tracker <https://github.com/NGSolve/ngsPETSc/issues>`__, describe the expected outcome and report the OS, the compiler, NGSolve/Netgen and PETSc version you are using.

Pull requests - patches, improvements, new features - are a fantastic
help. They should remain focused in scope and avoid containing unrelated commits.
**Please ask first** before embarking on any significant pull request.

Tips on opening a pull request:

1. `Fork <http://help.github.com/fork-a-repo/>`__. the project.

2. Create a branch and implement your feature.
   ::
   
        git checkout -b <your-feature-name>
   

3. Run the test suite by calling 
   ::

        make test test_mpi
   
   in your build directory. Consider adding new tests for your feature - have a look in the test folder.
   Keep in mind ngsPETSc test only tests NGSolve add-on features, while Firedrake can be found `here <https://github.com/firedrakeproject/firedrake/blob/master/tests/regression/test_netgen.py>`__ and `here <https://github.com/firedrakeproject/firedrake/blob/master/tests/multigrid/test_netgen_gmg.py>`__.
When you open a pull request all the testing is also carried out automatically for both Firedrake and Netgen by our `CI <https://github.com/NGSolve/ngsPETSc/blob/main/.github/workflows/ngsPETSc.yml>`__.

4. Once the implementation is done, use Git's
   `interactive rebase <https://help.github.com/articles/interactive-rebase>`__.
   feature to tidy up your commits.
   ::
   
        git rebase --interactive --fork-point main <your-feature-name> 
   

5. Push your topic branch up to your fork and `open a Pull Request <https://help.github.com/articles/using-pull-requests/>`__.

**IMPORTANT**: By submitting a patch, you agree to allow the project owners to license your work under the terms of the *GPL License*.

A code style is enforced using pylint. You can check your code passes the linting as follows:
::

    make lint

To actively discuss pull requests and issues you can use our `Discord channel <https://discord.gg/DpfXPdRSgV>`__.

Authors
----------

Jack Betteridge, Patrick E. Farrell, Stefano Zampini, Umberto Zerbinati

License
---------------

The package is released under the `MIT
License <https://opensource.org/licenses/MIT>`__.
