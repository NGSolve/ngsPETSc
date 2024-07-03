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
  - name: Joachim Schöberl
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

Combining advanced meshing techniques with robust solver capabilities is essential for solving difficult problems in computational science and engineering. This paper introduces ngsPETSc, a software built with petsc4py [@petsc4py] that seamlessly integrates the NETGEN mesher [@Netgen],the NGSolve finite element library [@NGSolve], and the PETSc toolkit [@PETSc]. ngsPETSc enables the use of NETGEN meshes and geometries in PETSc-based solvers, and provides NGSolve users access to the wide array of linear, non-linear solvers and time-steppers available in PETSc.

# Statement of Need

Efficiently solving large-scale partial differential equations (PDEs) on complex geometries is vital in scientific computing. PETSc, NETGEN, and NGSolve offer distinct functionalities: PETSc handles linear and nonlinear problems in a discretisation agnostic manner, NETGEN constructs meshes from constructive solid geometry (CSG) described with OpenCASCADE, and NGSolve offers a wide range of finite element discretisations. Integrating these tools with ngsPETSc promises to streamline simulation workflows and to enhance large-scale computing capabilities for challenging problems. This integration also facilitates seamless mesh exports from NETGEN to PETSc DMPlex, enabling simulations of intricate geometries and supporting advanced meshing techniques in other PETSc-based solvers, like Firedrake [@Firedrake].

In particular, by combining PETSc, NETGEN, and NGSolve within ngsPETSc the following new features are available:

- PETSc preconditioners, such as PETSc GAMG and PETSc ASM, can be used as building blocks inside the NGSolve preconditioning infrastructure;
- PETSc nonlinear solvers are available in NGSolve, in particular advanced line search and trust region Newton-based methods are available;
- NGSolve can now use PETSc linear solvers, including flexible and pipelined variants. In particular, Krylov solvers can also be used with NGSolve matrices stored in a matrix-free fashion as well as with NGSolve block matrices.
- high order meshes constructed in NETGEN together with adaptive mesh refinement and mesh hierarchies for geometric multigrid are now available in Firedrake [@Firedrake].

In conclusion, ngsPETSc is a lightweight, user-friendly interface that bridges the gap between NETGEN, NGSolve, and PETSc, building on top of petsc4py.
ngsPETsc aims to provide a comprehensive set of tools for solving complex PDEs on intricate geometries, enriching the already powerful capabilities of NETGEN, NGSolve, and Firedrake.

# Examples

In this section we provide a few examples of results that can be obtained using ngsPETSc.
We begin considering a simple Poisson problem on a unit square domain discretised with P2 finite elements and compare the performance of different solvers available in NGSolve via ngsPETSc. The result is shown in Table 1.

N. DoFs  | PETSc GAMG   | PETSc BDDC (N=2) | PETSc BDDC (N=4) | PETSc BDDC (N=6) | Element-wise BDDC* |
---------|--------------|------------------|------------------|------------------|--------------------|
116716   |35  (1.01e-05)|5 (9.46e-06)      |7 (1.27e-05)      |9 (5.75e-06)      |10 (2.40e-06)       |
464858   |69  (7.09e-06)|5 (8.39e-06)      |7 (1.27e-05)      |8 (8.19e-06)      |9 (6.78e-06)        |
1855428  |142 (3.70e-06)|        -         |8 (7.12e-06)      |9 (5.39e-06)      |10 (8.79e-06)       |

Table 1: In this table we report the number of degrees of freedom (DoFs) and the number of iterations required to solve the Poisson problem with different solvers. The numbers in parentheses are the relative residuals.

We then consider a more complex problem, the Oseen problem discretised using high-order Hood-Taylor elements (P4-P3) on a unit square domain. In particular we consider an augmented Lagrangian formulation to enforce the incompressibility constraint. We present the performance of a two level additive Schwarz preconditioner with vertex-patch smoothing as fine level correction. Such preconditioner was built using ngsPETSc, for simulating a lid driven cavity in presence of wind. The result for different Raynolds number are shown in Table 2.

Ref. Levels (N. DoFs) | Re=1e2       | Re=1e3      | Re=1e4      |
----------------------|--------------|-------------|-------------|
1 (83842)             |3  (9.52e-07) |4 (3.05e-06) |6 (2.56e-05) |
2 (334082)            |3  (6.09e-07) |4 (1.53e-06) |6 (8.85e-06) |
3 (1333762)           |3  (8.21e-07) |4 (3.84e-06) |6 (7.05e-06) |

Table 2: In this table we report the number of iterations required to solve the Oseen problem with different Reynolds numbers and different number of refinement levels, in parentheses we report the number of degrees of freedom (DoFs) on the finest level and the relative residuals.


