#Setting up basic python and to be root
FROM python:3.10-slim-buster
USER root
#Setting up system variable
ENV PETSC_DIR /root/petsc
ENV PETSC_ARCH linux_debug
ENV SLEPC_DIR /root/slepc
ENV SLEPC_ARCH linux_debug
ENV PYTHONPATH /root/petsc/linux_debug/lib:/root/slepc/linux_debug/lib
#Installing dependencies using aptitude
RUN apt-get update \
    && apt-get -y install git libopenmpi-dev build-essential cmake python3 python3-distutils python3-tk libpython3-dev libxmu-dev tk-dev tcl-dev g++ libglu1-mesa-dev liblapacke-dev libblas-dev liblapack-dev
#Installing python dependencies using pip
RUN pip install numpy cython mpi4py pytest pytest-mpi
#Configure PETSc
RUN cd ~ && git clone https://gitlab.com/petsc/petsc.git
RUN cd ~/petsc \
    && python configure --download-chaco \
    --download-cmake \
    --download-eigen \
    --with-openmpi=1 \
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
    && make \ 
    && ln -s "$(which mpiexec)" $PETSC_DIR/$PETSC_ARCH/bin/mpiexec
#Configure SLEPc
RUN cd ~ && git clone https://gitlab.com/slepc/slepc.git
RUN cd ~/slepc \
    && python configure --download-blopex \
    --with-slepc4py=1 \
    && make 
#Building ngsolve
RUN mkdir -p ~/ngsuite \
           && cd ~/ngsuite \
           && git clone https://github.com/NGSolve/ngsolve.git ngsolve-src \
           && cd ~/ngsuite/ngsolve-src \
           && git submodule update --init --recursive \
           && mkdir ~/ngsuite/ngsolve-build \
           && mkdir ~/ngsuite/ngsolve-install \
           && cd ~/ngsuite/ngsolve-build \
           && cmake -DCMAKE_INSTALL_PREFIX=~/ngsuite/ngsolve-install ~/ngsuite/ngsolve-src -DUSE_MPI=ON -DUSE_MPI4PY=ON -DUSE_OCC=ON -DBUILD_OCC=ON\
           && make && make install
#Adding NGS to PYTHONPATH
ENV PYTHONPATH /root/petsc/linux_debug/lib:/root/slepc/linux_debug/lib:/root/ngsuite/ngsolve-install/lib/python3.10/site-packages
