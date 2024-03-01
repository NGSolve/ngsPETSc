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
In the ever-evolving landscape of computational science and engineering, the synthesis of advanced meshing techniques and robust solver capabilities is fundamental to pushing the boundaries of simulation accuracy and efficiency. Addressing this need, we introduce ngsPETSc, a cutting-edge interface that seamlessly integrates the NETGEN mesher [@NETGEN], the versatile NGSolve finite element library [@NGSolve], and the high-performance Portable, Extensible Toolkit for Scientific Computation (PETSc)[@PETSc].

On the meshing front, ngsPETSc serves as a bridge between NETGEN and PETSc DMPlex, offering a sophisticated framework for the export of finely crafted meshes generated from constructive solid geometry (CSG) descriptions using OpenCASCADE. This integration extends the capabilities of mesh-based simulations, empowering researchers and practitioners to leverage intricate geometries and realize higher-order meshing strategies. In particular, ngsPETSc interface with Firedrake\ [@Firedrake] expands these possibilities, enabling the utilization of CSG-derived meshes while maintaining conformity to complex geometries and facilitating the construction of mesh hierarchies for geometric multigrid solvers.

On the solver front, ngsPETSc empowers NGSolve with access to the extensive suite of PETSc solvers through interfaces with PC, KSP, and SNES. This integration broadens the range of linear and nonlinear solvers available to NGSolve, providing a versatile toolkit for tackling a diverse array of complex problems.

This paper unfolds the capabilities and potential of ngsPETSc, contributing to the evolving landscape of computational simulations by seamlessly integrating advanced meshing and solver functionalities. In doing so, we will showcase the diverse range of problems and scenarios where ngsPETSc proves instrumental, from fluid dynamics to structural mechanics and beyond. 

# Statement of need


# References