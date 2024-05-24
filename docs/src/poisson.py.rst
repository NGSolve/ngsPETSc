Solving the Poisson using PETSc KSP
==============================================

In this tutorial we explore solving the Poisson problem with Dirichlet boundary conditions using `PETSc KSP`.
We begin by creating a discretisation of the Poisson problem using H1 elements, in particular we consider the usual variational formulation
.. math::

   \text{find } u\in H^1_0(\Omega) \text{ s.t. } a(u,v) := \int_{\Omega} \nabla u\cdot \nabla v \; d\vec{x} = L(v) := \int_{\Omega} fv\; d\vec{x}\qquad v\in H^1_0(\Omega).

Such a discretisation can easily be constructed using NGSolve as follows: ::

   from ngsolve import *
   import netgen.gui
   import netgen.meshing as ngm
   from mpi4py.MPI import COMM_WORLD

   if COMM_WORLD.rank == 0:
      mesh = Mesh(unit_square.GenerateMesh(maxh=0.2).Distribute(COMM_WORLD))
   else:
      mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
   fes = H1(mesh, order=3, dirichlet="left|right|top|bottom")
   u,v = fes.TnT()
   a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
   f = LinearForm(fes)
   f += 32 * (y*(1-y)+x*(1-x)) * v * dx

We now import from ngsPETSc the `KrylovSolver` object that exposes to NGSolve the `PETSc KSP` object, using petsc4py. 
The `PETSc KSP` provides a wide range of iterative solvers and preconditioners that can be used to solve a linear system of equations, i.e.
.. math::

   Au= b

where A is a `PETSc Mat`, u and b are `PETSc Vec`. More information on the `PETSc KSP` can be found `here <https://petsc.org/main/manual/ksp/>`__.
We then initialise a `KrylovSolver` passing: the NGSolve `BilinearForm` we want associated to the matrix of our linear system, the NGSolve `FiniteElementSpace` over which the bilinear form we are cosnidering is assembled and a dictionary `solverParameters` containing the PETSc command line flags used to configure the solver. 
Lastly we solve the lienar system passing to the method `solve` of the `KrylovSolver` a NGSolve `LinearForm` reppresenting the load vector we are consdiering and that is the right-hand side of the linear system. 
We first solve the lineary system by a direct solve perforemd using  `MUMPS <https://mumps-solver.org/index.php>`__. ::

   from ngsPETSc import KrylovSolver
   opts = {'ksp_type': 'preonly',
           'pc_type': 'lu',
           'pc_factor_mat_solver_type': 'mumps'}
   solver = KrylovSolver(a,fes, solverParameters=opts)
   gfu = solver.solve(f)
   Draw(gfu,mesh, "solution")

Lastly, we can inspect the PETSc KSP object using the method `view` of the `KrylovSolver` object. ::

   solver.ksp.view()

We can also can also use a Jacobi iteration to solve the linear system, setting a different `ksp_type`: ::

   opts = {'ksp_type': 'richardson',
           'ksp_richardson_scale':  1.0,
           'pc_type': 'jacobi'}
   solver = KrylovSolver(a,fes, solverParameters=opts)
   gfu = solver.solve(f)
   Draw(gfu,mesh, "solution")

Laslty, we show how to use `PETSc GAMG` as a preconditioner inside the a conjugated gradient solver. ::

   opts = {'ksp_type': 'cg',
           'pc_type': 'gamg'}
   solver = KrylovSolver(a,fes, solverParameters=opts)
   gfu = solver.solve(f)
