---
title: 'ngsPETSc: A coupling between NETGEN/NGSolve and PETSc'
tags:
  - PETSc
  - FEM
  - Meshing
authors:
  - name: Jack Betteridge
    orcid: 0000-0002-3919-8603
    equal-contrib: true
    affiliation: 1
  - name: Patrick E. Farrell
    orcid: 0000-0002-1241-7060
    equal-contrib: true
    affiliation: 3
  - name: Matthias Hochsteger
    orcid: 0009-0001-8842-3221
    equal-contrib: true
    affiliation: 2
  - name: Christopher Lackner
    orcid: 0009-0000-3448-3002
    equal-contrib: true
    affiliation: 2
  - name: Joachim Schöberl
    affiliation: 2,4
    equal-contrib: true 
    orcid: 0000-0002-1250-5087
  - name: Stefano Zampini
    orcid: 0000-0002-0435-0433
    equal-contrib: true 
    affiliation: 5
  - name: Umberto Zerbinati
    orcid: 0000-0002-2577-1106
    corresponding: true
    equal-contrib: true 
    affiliation: 2
affiliations:
 - name: Imperial College London, United Kingdom
   index: 1
 - name: CERBSim GmbH, Austria
   index: 2
 - name: University of Oxford, United Kingdom
   index: 3
 - name: TU Wien, Austria
   index: 4
 - name: King Abdullah University of Science and Technology, Saudi Arabia
   index: 5
date: 1 July 2024
bibliography: paper.bib
---

# Summary

Combining advanced meshing techniques with robust solver capabilities is essential for solving difficult problems in computational science and engineering. In recent years, various software packages have been developed to support the integration of meshing tools with finite element solvers. To mention a few, FreeFEM [@FreeFEM] includes built-in support for mesh generation, allowing users to create and manipulate meshes directly within the software. Similarly, deal.II [@dealII] provides a GridGenerator class for generating standard mesh geometries like grids and cylinders. Furthermore, deal.II can interface with OpenCASCADE [@OpenCASCADE] to refine existing grids while conforming to the geometry provided. Other finite element libraries, such as Firedrake [@Firedrake], DUNE-FEM [@DUNE], and FEniCSx [@dolfinX], rely on external tools like Gmsh [@GMSH] and Tetgen [@TETGEN] for mesh generation. This paper introduces ngsPETSc, software built with petsc4py [@petsc4py] that seamlessly integrates the NETGEN mesher [@Netgen], the NGSolve finite element library [@NGSolve], and the PETSc toolkit [@PETSc]. ngsPETSc enables the use of NETGEN meshes and geometries in solvers that use PETSc's DMPLEX [@DMPLEX], and provides NGSolve users access to the wide array of linear, nonlinear solvers, and time-steppers available in PETSc.

# Statement of Need

Efficiently solving large-scale partial differential equations (PDEs) on complex geometries is vital in scientific computing. PETSc, NETGEN, and NGSolve offer distinct functionalities: PETSc handles linear and nonlinear problems in a discretisation agnostic manner, NETGEN constructs meshes from constructive solid geometry (CSG) described with OpenCASCADE [@OpenCASCADE], and NGSolve offers a wide range of finite element discretisations. Integrating these tools with ngsPETSc promises to streamline simulation workflows and to enhance large-scale computing capabilities for challenging problems. This integration also facilitates seamless mesh exports from NETGEN to PETSc DMPlex, enabling simulations of complex geometries and supporting advanced meshing techniques in other PETSc-based solvers that employ DMPLEX. We illustrate this with the Firedrake finite element system [@Firedrake].

In particular, by combining PETSc, NETGEN, and NGSolve within ngsPETSc the following new features are available:

- PETSc Krylov solvers, including flexible and pipelined variants, are available in NGSolve. They can be used both with NGSolve matrix-free operators and NGSolve block matrices;
- PETSc preconditioners can be used as components within the NGSolve preconditioning infrastructure;
- PETSc nonlinear solvers are available in NGSolve, including advanced line search and trust region Newton-based methods;
- high order meshes constructed in NETGEN are now available in Firedrake [@Firedrake], enabling adaptive mesh refinement and geometric multigrid on hierarchies of curved meshes.

In conclusion, ngsPETSc is a lightweight, user-friendly interface that bridges the gap between NETGEN, NGSolve, and PETSc, building on top of petsc4py.
ngsPETsc aims to assist with the solution of challenging PDEs on complex geometries, enriching the already powerful capabilities of NETGEN, NGSolve, PETSc, and Firedrake.

# Examples

In this section we provide a few examples of results that can be obtained using ngsPETSc.
We begin by considering a simple primal Poisson problem on a unit square domain discretised with conforming $P_2$ finite elements and compare the performance of different solvers newly available in NGSolve via ngsPETSc. In particular, we consider PETSc's algebraic multigrid algorithm GAMG [@PETScGAMG], PETSc's domain decomposition BDDC algorithm [@PETScBDDC], NGSolve's own implementation of element-wise BDDC, the Hypre algebraic multigrid algorithm [@hypre] and the ML algebraic multigrid algorithm [@ml], each combined with the conjugate gradient method. 
Other than the elementwise BDDC preconditioner, these preconditioners were not previously available in NGSolve. The results are shown in Table 1 and the full example, with more details, can be found in the [ngsPETSc documentation](https://ngspetsc.readthedocs.io/en/latest/PETScKSP/poisson.py.html).
All the preconditioners considered exhibit robust conjugate gradient iteration counts as we refine the mesh for a $P_1$ discretisation, but out-of-the-box only BDDC type preconditioners are robust as we refine the mesh for a $P_2$ discretisation. A possible remedy for this issue is discussed in the [ngsPETSc documentation](https://ngspetsc.readthedocs.io/en/latest/PETScPC/poisson.py.html).

\# DoFs  | PETSc GAMG   | HYPRE | ML  | PETSc BDDC* | Element-wise BDDC** |
---------|--------------|-------|-----|------------|--------------------|
116716   |35            | 36    | 31  |9           |10                  |
464858   |69            | 74    | 63  |8           |9                   |
1855428  |142           | 148   | 127 |9           |10                  |

Table 1: The number of degrees of freedom (DoFs) and the number of iterations required to solve the Poisson problem with different solvers. Each row corresponds to a level of uniform refinement.  The conjugate gradient solve was terminated when the residual norm decreased by six orders of magnitude. *We choose to use PETSc BDDC with six subdomains. **Element-wise BDDC is a custom implementation of BDDC in NGSolve.

We next consider the Oseen problem, i.e.
$$
\nu\Delta \vec{u} +\vec{b}\cdot \nabla\vec{u} - \nabla p = \vec{f},
\\ \quad \nabla \cdot \vec{u} = 0,
$$
We discretise this problem using high-order Hood-Taylor elements ($P_4$-$P_3$) on a unit square domain [@HT; @Boffi]. We employ an augmented Lagrangian formulation to better enforce the incompressibility constraint. We present the performance of GMRES [@GMRES] preconditioned with a two level additive Schwarz preconditioner with vertex-patch smoothing as fine level correction [@BenziOlshanskii; @FarrellEtAll]. This preconditioner was built using ngsPETSc. The result for different viscosities $\nu$ are shown in Table 2, exhibiting reasonable robustness as the viscosity (and hence Reynolds number) changes. The full example, with more details, can be found in the [ngsPETSc documentation](https://ngspetsc.readthedocs.io/en/latest/PETScPC/oseen.py.html).

\# refinements (\# DoFs) | $\nu=10^{-2}$|$\nu=10^{-3}$|$\nu=10^{-4}$|
----------------------|--------------|-------------|-------------|
1 (83842)             |3             |4            |6            |
2 (334082)            |3             |4            |6            |
3 (1333762)           |3             |4            |6            |

Table 2: The number of iterations required to solve the Oseen problem with different viscosities and different refinement levels. In parentheses we report the number of degrees of freedom (DoFs) on the finest level. The GMRES iteration was terminated when the residual norm decreased by eight orders of magnitude. 

Figure 1 shows a simulation of a hyperelastic beam, solved with PETSc nonlinear solvers; the line search algorithms in PETSc solve this straightforwardly, but an undamped Newton iteration does not converge.
Figures 2 and 3 show simulations in Firedrake that were not previously possible. Figure 2 shows a high-order NETGEN mesh employed for the simulation of a Navier-Stokes flow past a cylinder, while Figure 3 shows adaptive mesh refinement for a Poisson problem on an L-shaped domain. The adaptive procedure achieves the optimal complexity of error with degree of freedom count, as expected [@stevenson2006].


![A hyperelastic beam deformed by fixing one end and applying a twist at the other end. The colouring corresponds to the deviatoric von Mises stress experienced by the beam. The beam is discretised with $P_3$ finite elements and the nonlinear problem is solved using PETSc SNES. The full example, with more details, can be found in the [ngsPETSc documentation](https://ngspetsc.readthedocs.io/en/latest/PETScSNES/hyperelasticity.py.html).](figures/hyperelastic.png)


![Flow past a cylinder. The Navier-Stokes equations are discretised on a NETGEN high-order mesh with Firedrake. We use high-order Taylor-Hood elements ($P_4$-$P_3$) and a vertex-patch smoother as fine level correction in a two-level additive Schwarz preconditioner, [@BenziOlshanskii; @FarrellEtAll]. The full example, with more details, can be found in [ngsPETSc documentation](https://github.com/NGSolve/ngsPETSc). On the right a zoom near the cylinder shows the curvature of the mesh.](figures/flow_past_a_cylinder.png)


![An adaptive scheme applied to the Poisson problem on an L-shaped domain. The domain is discretised using $P_1$ finite elements and the adaptive mesh refinement is driven by a Babuška-Rheinboldt error estimator [@BabuskaRheinboldt]. The adaptive procedure delivers optimal scaling of the energy norm of the error in terms of the number of degrees of freedom. The full example, with more details, can be found in the [ngsPETSc documentation](https://ngspetsc.readthedocs.io/en/latest/utils/firedrake/lomesh.py.html).](figures/adaptivity.png)


More examples can be found in the documentation of ngsPETSc manual [@manual].
