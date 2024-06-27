---
title: 'ngsPETSc: A coupling between NETGEN/NGSolve and PETSc'
tags:
  - PETSc
  - FEM
  - Meshing
authors:
  - name: Patrick E. Farrell
    orcid: 0000-0002-1241-7060
    equal-contrib: true
    affiliation: 1
  - name: Joachim Schoberl
    affiliation: 2
    equal-contrib: true 
    orcid: 0000-0002-1250-5087
  - name: Stefano Zampini
    orcid: 0000-0002-0435-0433
    equal-contrib: true 
    affiliation: 3
  - name: Umberto Zerbinati
    orcid: 0000-0002-2577-1106
    corresponding: true
    equal-contrib: true 
    affiliation: 1
affiliations:
 - name: University of Oxford, United Kingdom
   index: 1
 - name: TU Wien, Austria
   index: 2
 - name: King Abdullah University of Science and Technology
   index: 3
date: 1 July 2024
bibliography: paper.bib
---

# Summary

In the realm of computational science and engineering, combining advanced meshing techniques with robust solver capabilities is essential for improving simulation accuracy and efficiency. This paper introduces ngsPETSc, an interface build on top of petsc4py [@petsc4py] that seamlessly integrates the NETGEN mesher [@Netgen], NGSolve finite element library,[@NGSolve] and PETSc toolkit[@PETSc]. ngsPETSc enables the use of NETGEN meshes and geometries in software based on PETSc DMPlex. Additionally, ngsPETSc provides NGSolve users access the wide array of linear, non-linear solvers and time-steppers available in PETSc.

# Steatment of Need

Efficiently solving large-scale Partial Differential Equations (PDEs) on complex geometries is vital in scientific computing. PETSc, NETGEN, and NGSolve offer distinct functionalities: PETSc handles linear and nonlinear problems in a discretisation agnostic manner, NETGEN constructs meshes from constructive solid geometry (CSG) descripted with OpenCASCADE, and NGSolve offers a wide range of finite element discretizations to the user. Integrating these tools into ngsPETSc promises to streamline workflows and enhance capabilities in scientific computing. This integration also facilitates seamless mesh exports from NETGEN to PETSc DMPlex, enabling simulations of intricate geometries and supporting advanced meshing techniques. Moreover, ngsPETSc equips NGSolve with PETSc's extensive suite of solvers via interfaces to PETSc PC, KSP, and SNES. This broadens the range of solvers for tackling diverse computational challenges.

In particular, by combining PETSc, NETGEN, and NGSolve within ngsPETSc the following features are aveilable:

- high order meshes constructed in NETGEN together with adaptive mesh refinement are now available in Firedrake [@Firedrake];
- mesh hierarchies generated from OpenCASCADE geometries can be used in Firedrake;
- PETSc preconditers, such as PETSc GAMG and PETSc ASM, can be used as building blocks inside NGSolve preconditioning infrastructure;
- PETSc nonlinear solvers are available in NGSolve, in particular second order methods are aveilable thanks to NGSolve ability to compute the Jacobian for any discretization;
- NGSolve can now use PETSc linear solvers, such as PETSc CG, PETSc GMRES, PETSc BiCGStab, etc. In particular, Krylov solvers can also be used with NGSolve matrices stored in a matrix free fashion as well as with NGSolve block matrices.
- NGSolve can now use PETSc time integrators, such as PETSc TS, to solve time dependent problems.

In conclusion, ngsPETSc is a lightweight, user-friendly interface that bridges the gap between NETGEN, NGSolve, and PETSc, building on top of petsc4py.
ngsPETsc aims to provide a comprehensive set of tools for solving complex PDEs on intricate geometries, enriching the already powerful capabilities of NETGEN, NGSolve, and Firedrake.