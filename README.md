# ngsPETSc

ngsPETSc is an interface between PETSc and NGSolve/NETGEN that enables the use of NETGEN meshes and geometries in PETSc-based solvers while providing NGSolve users access to the wide array of linear, nonlinear solvers, and time-steppers available in PETSc.

[![ngsPETSc](https://github.com/UZerbinati/ngsPETSc/actions/workflows/ngsPETSc.yml/badge.svg)](https://github.com/UZerbinati/ngsPETSc/actions/workflows/ngsPETSc.yml)
[![Documentation Status](https://readthedocs.org/projects/ngspetsc/badge/?version=latest)](https://ngspetsc.readthedocs.io/en/latest/?badge=latest)

## Installation
If you already have NGSolve (with MPI support) and PETSc installed, you can install ngsPETSc via pip:
```bash

    git clone https://github.com/UZerbinati/ngsPETSc.git
    cd ngsPETSc
    pip install .
```
Alternatively, you can also build PETSc, SLEPc, and NGSolve from source following the instructions in the [documentation](https://ngspetsc.readthedocs.io/en/latest/install.html).

## Getting started

To get started with ngsPETSc, check out the [documentation](https://ngspetsc.readthedocs.io/en/latest/).
To test the installation, you can run the tests in the `tests` folder, via the Makefile in the root directory of the repository:
```bash
    make test
```
