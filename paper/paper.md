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

Efficiently solving large-scale partial differential equations (PDEs) on complex geometries is vital in scientific computing. PETSc, NETGEN, and NGSolve offer distinct functionalities: PETSc handles linear and nonlinear problems in a discretisation agnostic manner, NETGEN constructs meshes from constructive solid geometry (CSG) described with OpenCASCADE[@OpenCASCADE], and NGSolve offers a wide range of finite element discretisations. Integrating these tools with ngsPETSc promises to streamline simulation workflows and to enhance large-scale computing capabilities for challenging problems. This integration also facilitates seamless mesh exports from NETGEN to PETSc DMPlex, enabling simulations of intricate geometries and supporting advanced meshing techniques in other PETSc-based solvers, like Firedrake [@Firedrake].

In particular, by combining PETSc, NETGEN, and NGSolve within ngsPETSc the following new features are available:

- PETSc Krylov solvers, including flexible and pipelined variants, are aveilable in NGSolve. In particular, they can also be used with NGSolve matrices stored in a matrix-free fashion as well as with NGSolve block matrices.
- PETSc preconditioners, can be used as building blocks inside the NGSolve preconditioning infrastructure;
- PETSc nonlinear solvers are available in NGSolve, in particular advanced line search and trust region Newton-based methods are available. A use case of PETSc nonlinear solvers for the simulation of a hypetelastic beam is shown in Figure 1.
- high order meshes constructed in NETGEN together with adaptive mesh refinement and mesh hierarchies for geometric multigrid are now available in Firedrake [@Firedrake]. A use case of high-order mesh in Firedrake for the simulation of a flow past a cylinder is shown in Figure 2. While a use case of adaptive mesh refinement for the simulation of a Poisson problem on a Pacman domain is shown in Figure 3.

In conclusion, ngsPETSc is a lightweight, user-friendly interface that bridges the gap between NETGEN, NGSolve, and PETSc, building on top of petsc4py.
ngsPETsc aims to provide a comprehensive set of tools for solving complex PDEs on intricate geometries, enriching the already powerful capabilities of NETGEN, NGSolve, and Firedrake.

# Examples

In this section we provide a few examples of results that can be obtained using ngsPETSc.
We begin considering a simple Poisson problem on a unit square domain discretised with P2 finite elements and compare the performance of different solvers available in NGSolve via ngsPETSc. The result is shown in Table 1 and the full example, with more details, can be found in the [ngsPETSc documentation](https://ngspetsc.readthedocs.io/en/latest/PETScKSP/poisson.py.html).

N. DoFs  | [@PETScGAMG]  | [@PETScBDDC] (N=2) | [@PETScBDDC] (N=4) | [@PETScBDDC] (N=6) | Element-wise BDDC* |
---------|--------------|------------------|------------------|------------------|--------------------|
116716   |35  (1.01e-05)|5 (9.46e-06)      |7 (1.27e-05)      |9 (5.75e-06)      |10 (2.40e-06)       |
464858   |69  (7.09e-06)|5 (8.39e-06)      |7 (1.27e-05)      |8 (8.19e-06)      |9 (6.78e-06)        |
1855428  |142 (3.70e-06)|        -         |8 (7.12e-06)      |9 (5.39e-06)      |10 (8.79e-06)       |

Table 1: In this table we report the number of degrees of freedom (DoFs) and the number of iterations required to solve the Poisson problem with different solvers. The numbers in parentheses are the relative residuals. *Element-wise BDDC is a custom implementation of BDDC preconditioner in NGSolve.

We then consider the Oseen problem, i.e.
$$
\nu\Delta \vec{u} +\vec{b}\cdot \nabla\vec{u} - \nabla p = \vec{f}
\\ \quad \nabla \cdot \vec{u} = 0,
$$
We discretise such problem using high-order Hood-Taylor elements (P4-P3) on a unit square domain. In particular we consider an augmented Lagrangian formulation to enforce the incompressibility constraint. We present the performance of a two level additive Schwarz preconditioner with vertex-patch smoothing as fine level correction [@BenziOlshanskii]. Such preconditioner was built using ngsPETSc, for simulating a lid driven cavity in presence of wind. The result for different Raynolds number are shown in Table 2 and the full example, with more details, can be found in the [ngsPETSc documentation](https://ngspetsc.readthedocs.io/en/latest/PETScPC/oseen.py.html).

Ref. Levels (N. DoFs) | $\nu=10^{-2}$|$\nu=10^{-3}$|$\nu=10^{-4}$|
----------------------|--------------|-------------|-------------|
1 (83842)             |3  (9.52e-07) |4 (3.05e-06) |6 (2.56e-05) |
2 (334082)            |3  (6.09e-07) |4 (1.53e-06) |6 (8.85e-06) |
3 (1333762)           |3  (8.21e-07) |4 (3.84e-06) |6 (7.05e-06) |

Table 2: In this table we report the number of iterations required to solve the Oseen problem with different Reynolds numbers and different number of refinement levels, in parentheses we report the number of degrees of freedom (DoFs) on the finest level and the relative residuals.


![An hyperelastic beam deformed by keeping one end fixed and applying a twist at the other end. The coloring corresponds to the deviatoric von Mises stress experienced by the beam. The beam is discretised with P3 finite elements and the non-linear problem is solved using SNES. The full example, with more details, can be found in the [ngsPETSc documentation](https://ngspetsc.readthedocs.io/en/latest/PETScSNES/hyperelasticity.py.html).](figures/hyperelastic.png)


![On the right a flow past a cylinder simulation, discretised using a Netgen high-order mesh and Firedrake. In particular we use high-order Taylor-Hood elements (P4-P3) and a vertex-patch smoother as fine level correction in a two-level additive Schwarz preconditioner, [@BenziOlshanskii]. The full example, with more details, can be found in [ngsPETSc repo](https://github.com/NGSolve/ngsPETSc). On the left a zoom near the cylinder to show that the mesh is high-order.](figures/flow_past_a_cylinder.png)


![An adaptive scheme applied to the Poisson problem on a Pacman domain. The domain is discretised using P1 finite elements and the adaptive mesh refinement is driven by a Babuška-Rheinboldt error estimator [@BabuskaRheinboldt]. The full example, with more details, can be found in the [ngsPETSc documentation](https://ngspetsc.readthedocs.io/en/latest/utils/firedrake/lomesh.py.html).](figures/adaptive.png)


More example can be found in the documentation of ngsPETSc manual [@manual].