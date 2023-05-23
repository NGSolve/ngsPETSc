#Setting up basic python and to be root
FROM python:3.10-slim-buster
USER root
#Setting up system variable
ENV PETSC_DIR /root/petsc
ENV PETSC_ARCH linux_debug
ENV PYTHONPATH /root/petsc/linux_debug/lib
#Installing dependencies using aptitude
RUN apt-get update \
    && apt-get -y install git libopenmpi-dev build-essential cmake python3 python3-distutils python3-tk libpython3-dev libxmu-dev tk-dev tcl-dev g++ libglu1-mesa-dev liblapacke-dev
#Installing python dependencies using pip
RUN pip install numpy cython mpi4py
#Configure PETSc
RUN cd ~ && git clone https://gitlab.com/petsc/petsc.git
RUN cd ~/petsc \
    && python configure --download-chaco \
    --download-cmake \
    --download-eigen \
    --with-openmpi=1 \
    --download-hdf5 \
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
    && make 
#Building ngsolve
RUN export BASEDIR=~/ngsuite \
           && mkdir -p $BASEDIR \
           && cd $BASEDIR \
           && git clone https://github.com/NGSolve/ngsolve.git ngsolve-src \
           && cd $BASEDIR/ngsolve-src \
           && git submodule update --init --recursive \
           && mkdir $BASEDIR/ngsolve-build \
           && mkdir $BASEDIR/ngsolve-install \
           && cd $BASEDIR/ngsolve-build \
           && cmake -DCMAKE_INSTALL_PREFIX=${BASEDIR}/ngsolve-install ${BASEDIR}/ngsolve-src -DUSE_MPI=ON \
           && make && make install
