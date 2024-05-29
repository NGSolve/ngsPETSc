import setuptools, os

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

if 'NGSPETSC_NO_INSTALL_REQUIRED' in os.environ:
    install_requires = []
elif 'NGS_FROM_SOURCE' in os.environ:
    install_requires = [
        'petsc4py',
        'mpi4py',
        'numpy',
        'pytest', #For testing
        'pylint', #For formatting
        ]
else:
    install_requires=[
        'netgen-mesher',
        'ngsolve',
        'petsc4py',
        'mpi4py',
        'pytest', #For testing
        'pylint', #For formatting
    ]

setuptools.setup(
    name = "ngsPETSc",
    version = "0.0.5",
    author = "Umberto Zerbinati",
    author_email = "umberto.zerbinati@maths.ox.ac.uk",
    description = "NGSolve/Netgen interface to PETSc.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "",
    project_urls = {
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["ngsPETSc", "ngsPETSc.utils", "ngsPETSc.utils.firedrake"],
    python_requires = ">=3.8",
    install_requires=install_requires

)
