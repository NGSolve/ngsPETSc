---
title: 'ngsPETSc: Yet an other NGSolve/Netgen -- PETSc interface'
tags:
  - Python
  - PDE
  - FEM
  - Mesh
authors:
  - name: Patrick E. Farrell
    orcid: 0000-0002-1241-7060
    equal-contrib: true
    affiliation: 1
  - name: Joachim Schöberl
    equal-contrib: true
    affiliation: 2
  - name: Stefano Zampini
    orcid: 0000-0002-0435-0433
    equal-contrib: true
    affiliation: 3
  - name: Umberto Zerbinati
    orcid: 0000-0002-2577-1106
    equal-contrib: true
    corresponding: true
    affiliation: 1
affiliations:
 - name: University of Oxford, United Kingdom
   index: 1
 - name: TU Wien, Austria
   index: 2
 - name: King Abdullah University of Science and Technology, Saudi Arabia
   index: 3
date: 13 August 2017
bibliography: paper.bib

---

# Summary

There are few common ingredients in any Finite Element (FE) simulation, among these common ingredients there are mesh generation and efficient linear or non-linear solves.
Furthermore in recent years that have been an increasing focus in the development of finite element softwares both as useful research tools and also to solve real-life applications, just to mention a few: `Deal II` [@DealII], `Dune` [@DuneFEM; @DuneVEM], `FEniCSx` [@Basix; @UFL], `FEniCS` [@FEniCS], `Firedrake` [@Firedrake] and `NGSolve` [@NGSolve].
Among the previously mentioned software packages there those whose strength lies in their intimate nature with a liner and non-linear solver such as `Firedrake`, `FEniCS` and `FEniCSx`, which relies on `PETSc` [@PETSc] as solver.
Other software packages such as `NGSolve` found their strength in their relation with the Netgen [@Netgen] mesh generator.
`NGSPETSc` is a Python software package that brings these two ingredients by improving mesh generation capabilities of `PETSc` based solvers, such as `Firedrake` or `FEniCSx`, while at the same giving access to `PETSc` preconditioner together with linear and non-linear solvers to NGSolve.

# Statement of need
`ngsPETSc` provides an interface between the `NGSolve`/`Netgen` software family to PETSc. We begin discussing the `Netgen` to `PETSc` interface, which to the best of our knowledge is a major difference with any previous `NGSolve`-`PETSc` interface. In particular, the `Netgen` to `PETSc` interface offers access to Netgen meshes exported in the form a PETSc `DMPlex` [@DMPlex]. Accessing `Netgen` meshes as PETSc `DMPlex` the following new features are available in `Firedrake` and `FEniCSx`:
- `Netgen` linear mesh can be used in both `Firedrake` and `FEniCSx` to solve all sorts of partial differential equations (PDEs). In particular, both two and three dimensional constructive solid geometries can be used to describe the domain where we aim to solve a PDE. Furthermore, thanks to new `Netgen` support for `OpenCascade` it is possible to describe the domain of solution of a PDE via the `OpenCascade` framework.
- Anisotropic refined mesh generated via `Netgen` can be used in `Firedrake` and `FEniCSx`.
- `Netgen` high order mesh can used in `Firedrake`. In particular the `OpenCascade` framework can be used to described arbitrary curved domains. Futhermore, curved Alfeld splits mesh can be constructed and used in `Firedrake`.
- Adaptive mesh refinement is now available in `Firedrake` thanks to `Netgen` support for adaptivity.


# References