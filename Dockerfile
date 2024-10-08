#Setting up basic python and to be root
FROM python:3.10-slim
USER root
#Setting up system variable
ENV PETSC_DIR /root/petsc
ENV PETSC_ARCH linux_debug
ENV SLEPC_DIR /root/slepc
ENV SLEPC_ARCH linux_debug
#Installing dependencies using aptitude
RUN apt-get update \
    && apt-get -y install git libopenmpi-dev build-essential cmake wget libssl-dev python3 python3-distutils python3-tk libpython3-dev libxmu-dev tk-dev tcl-dev g++ libglu1-mesa-dev liblapacke-dev libblas-dev liblapack-dev
#RUN apt-get update \
#    && apt-get -y install libocct-data-exchange-dev libocct-draw-dev occt-misc
#Building cmake
RUN cd ~ && wget https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6.tar.gz \
    && tar -zxvf cmake-3.27.6.tar.gz \
    && cd cmake-3.27.6 \
    && ./configure \
    && make -j 2 \
    && make install
#Installing python dependencies using pip
RUN pip install numpy scipy cython mpi4py pytest pytest-mpi
#Configure PETSc
RUN cd ~ && git clone https://gitlab.com/petsc/petsc.git
RUN cd ~/petsc \
    && python configure --download-chaco \
    --with-openmpi=1 \
    --download-hypre \
    --download-metis \
    --download-parmetis \
    --download-mumps \
    --download-scalapack \
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
ENV LD_LIBRARY_PATH /root/petsc/linux_debug/lib
RUN pip install netgen-occt-devel netgen-occt
RUN mkdir -p ~/ngsuite \
           && cd ~/ngsuite \
           && git clone https://github.com/NGSolve/ngsolve.git ngsolve-src \
           && cd ~/ngsuite/ngsolve-src \
           && git submodule update --init --recursive \
           && mkdir ~/ngsuite/ngsolve-build \
           && mkdir ~/ngsuite/ngsolve-install \
           && cd ~/ngsuite/ngsolve-build \
           && cmake -DCMAKE_INSTALL_PREFIX=~/ngsuite/ngsolve-install ~/ngsuite/ngsolve-src -DUSE_MPI=ON -DBUILD_OCC=OFF\
           && make && make install
#Adding NGS to PYTHONPATH
ENV PYTHONPATH /root/petsc/linux_debug/lib:/root/slepc/linux_debug/lib:/root/ngsuite/ngsolve-install/lib/python3.10/site-packages
