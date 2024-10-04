import setuptools, os

if 'NGSPETSC_NO_INSTALL_REQUIRED' in os.environ:
    install_requires = []
elif 'NGS_FROM_SOURCE' in os.environ:
    install_requires = [
        'petsc4py',
        'mpi4py',
        'numpy',
        'scipy',
        'pytest', #For testing
        'pylint', #For formatting
        ]
else:
    install_requires=[
        'netgen-mesher',
        'ngsolve',
        'petsc4py',
        'mpi4py',
        'scipy',
        'pytest', #For testing
        'pylint', #For formatting
    ]

setuptools.setup(
    install_requires=install_requires,
    packages=["ngsPETSc", "ngsPETSc.utils", "ngsPETSc.utils.firedrake", "ngsPETSc.utils.ngs"]
)
