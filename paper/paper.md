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
 - name: King Abdullah University of Science and Technology, Saudi Arabia
   index: 3
date: 1 July 2024
bibliography: paper.bib
---

# Summary

Combining advanced meshing techniques with robust solver capabilities is essential for solving difficult problems in computational science and engineering. This paper introduces ngsPETSc, software built with petsc4py [@petsc4py] that seamlessly integrates the NETGEN mesher [@Netgen], NGSolve finite element library [@NGSolve], and PETSc toolkit [@PETSc]. ngsPETSc enables the use of NETGEN meshes and geometries in PETSc-based solvers, and provides NGSolve users access to the wide array of linear, non-linear solvers and time-steppers available in PETSc.

# Statement of Need

Efficiently solving large-scale partial differential equations (PDEs) on complex geometries is vital in scientific computing. PETSc, NETGEN, and NGSolve offer distinct functionalities: PETSc handles linear and nonlinear problems in a discretisation agnostic manner, NETGEN constructs meshes from constructive solid geometry (CSG) described with OpenCASCADE, and NGSolve offers a wide range of finite element discretisations to the user. Integrating these tools with ngsPETSc promises to streamline simulation workflows and enhance capabilities in scientific computing. This integration also facilitates seamless mesh exports from NETGEN to PETSc DMPlex, enabling simulations of intricate geometries and supporting advanced meshing techniques in other PETSc-based solvers, like Firedrake [@Firedrake]. Moreover, ngsPETSc equips NGSolve with PETSc's extensive suite of solvers via interfaces to the PETSc PC, KSP, and SNES objects. This broadens the range of solvers for tackling diverse computational challenges.

In particular, by combining PETSc, NETGEN, and NGSolve within ngsPETSc the following new features are available:

- PETSc preconditioners, such as PETSc GAMG and PETSc ASM, can be used as building blocks inside the NGSolve preconditioning infrastructure;
- PETSc nonlinear solvers are available in NGSolve, in particular advanced line search and trust region Newton-based methods are available thanks to NGSolve's ability to compute the Jacobian for any discretisation;
- NGSolve can now use PETSc linear solvers, including flexible and pipelined variants. In particular, Krylov solvers can also be used with NGSolve matrices stored in a matrix-free fashion as well as with NGSolve block matrices.
- NGSolve can now use PETSc time integrators, such as PETSc TS, to flexibly switch between algorithms for solving time-dependent problems;
- high order meshes constructed in NETGEN together with adaptive mesh refinement and mesh hierarchies for geometric multigrid are now available in Firedrake [@Firedrake].

In conclusion, ngsPETSc is a lightweight, user-friendly interface that bridges the gap between NETGEN, NGSolve, and PETSc, building on top of petsc4py.
ngsPETsc aims to provide a comprehensive set of tools for solving complex PDEs on intricate geometries, enriching the already powerful capabilities of NETGEN, NGSolve, and Firedrake.
