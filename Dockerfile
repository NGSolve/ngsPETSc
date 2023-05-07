#Setting up basic python and to be root
FROM python:3.10-slim-buster
USER root
#Setting up system variable
ENV PETSC_DIR /root/petsc
ENV PETSC_ARCH linux_debug
ENV PYTHONPATH /root/petsc/linux_debug/lib
#Installing dependencies using aptitude
RUN apt-get update \
    && apt-get -y install git libopenmpi-dev build-essential cmake
#Installing python dependencies using pip
RUN pip install numpy cython
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
#Installing petsc4py
#RUN cd ~/petsc/src/binding/petsc4py && pip install . \
