import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "ngsPETSc",
    version = "0.0.1",
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
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.10",
    install_requires=[
        'ngsolve',
        'petsc4py',
        'pytest', #For testing
        'pylint', #For formatting
    ]

)
