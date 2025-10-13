# ngsPETSc
[![ngsPETSc](https://github.com/NGSolve/ngsPETSc/actions/workflows/ngsPETSc.yml/badge.svg)](https://github.com/NGSolve/ngsPETSc/actions/workflows/ngsPETSc.yml)
[![Documentation Status](https://readthedocs.org/projects/ngspetsc/badge/?version=latest)](https://ngspetsc.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07359/status.svg)](https://doi.org/10.21105/joss.07359)

ngsPETSc is an interface between PETSc and NGSolve/NETGEN that enables the use of NETGEN meshes and geometries in PETSc-based solvers while providing NGSolve users access to the wide array of linear, nonlinear solvers, and time-steppers available in PETSc.

## Installation
ngsPETSc is available on [PyPI](https://pypi.org/project/ngsPETSc/).
If you have PETSc installed be sure to set the `PETSC_DIR` and `PETSC_ARCH` environment variables to the required values.
You can install by running:
```bash
pip install ngsPETSc
```

## Getting started
To get started with ngsPETSc, check out the [documentation](https://ngspetsc.readthedocs.io/en/latest/).

## Development
If you already have NGSolve (with MPI support) and PETSc installed, you can install ngsPETSc via pip:
```bash
git clone https://github.com/NGSolve/ngsPETSc.git
pip install ./ngsPETSc
```
Alternatively, you can also build PETSc, SLEPc, and NGSolve from source following the instructions in the [documentation](https://ngspetsc.readthedocs.io/en/latest/install.html).

### Testing
To test the installation, you can run the tests in the `tests` folder, via the Makefile in the root directory of the repository:
```bash
make test
```
