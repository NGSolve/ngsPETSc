.. ngsPETSc documentation master file, created by
   sphinx-quickstart on Fri Jul  7 04:06:25 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ngsPETSc's documentation!
====================================

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   
   install
   autoapi/ngsPETSc/index

.. toctree::
   maxdepth: 1
   :caption: PETSc Vec and PETSc Mat

   PETScBasic/pinvt.py

.. toctree::
   :maxdepth: 1
   :caption: PETSc KSP

   PETScKSP/poisson.py
   PETScKSP/elasticity.py

.. toctree::
   :maxdepth: 1
   :caption: PETSc PC

   PETScPC/poisson.py
   PETScPC/stokes.py
   PETScPC/oseen.py

.. toctree::
   :maxdepth: 1
   :caption: PETSc SNES

   PETScSNES/hyperelasticity.py

.. toctree::
   :maxdepth: 1
   :caption: SLEPc EPS

   SLEPcEPS/poisson.py

.. toctree::
   :maxdepth: 1
   :caption: Firedrake and FEniCSx

   utils/firedrake/lomesh.py
   utils/firedrake/homesh.py
   utils/firedrake/surfaces.py
   FEniCSx-Netgen interface via ngsPETSc <https://jsdokken.com/dolfinx-tutorial/chapter2/amr.html>
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
