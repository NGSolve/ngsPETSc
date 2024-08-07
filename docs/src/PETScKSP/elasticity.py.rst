Solving linear elasticity with a near nullspace
==============================================

In this tutorial, we explore a linear elasticity discretisation.
As we did in :doc:`poisson.py`, we will solve the linear system arising from the discretisation using a `PETSc KSP`.
We begin by creating a discretisation for the weak formulation of linear elasticity with Lamé coefficients :math:`\mu` and :math:`\lambda`, i.e. find :math:`\vec{u}\in [H^1_0(\Omega)]^d` such that

.. math::

   a(u,v) := 2\mu \int_{\Omega} \epsilon(\vec{u}) : \epsilon(\vec{v}) \; d\vec{x} + \lambda \int_\Omega (\nabla \cdot \vec{u})\; d\vec{x} = L(v) := \int_{\Omega} fv\; d\vec{x}\qquad \vec{v}\in [H^1_0(\Omega)]^d.

We can easily discretise this problem using NGSolve: ::

   from ngsolve import *
   import netgen.gui
   import netgen.meshing as ngm
   from mpi4py.MPI import COMM_WORLD

   mesh = Mesh(unit_square.GenerateMesh(maxh=0.1,comm=COMM_WORLD))

   E, nu = 210, 0.2
   mu  = E / 2 / (1+nu)
   lam = E * nu / ((1+nu)*(1-2*nu))

   def Stress(strain):
      return 2*mu*strain + lam*Trace(strain)*Id(2)

   fes = VectorH1(mesh, order=1, dirichlet="left")
   u,v = fes.TnT()

   a = BilinearForm(InnerProduct(Stress(Sym(Grad(u))), Sym(Grad(v)))*dx)
   a.Assemble()

   force = CF( (0,1) )
   f = LinearForm(force*v*ds("right")).Assemble()

We begin solving the linear system using PETSc's own implementation of an algebraic multigrid preconditioner. ::

   from ngsPETSc import KrylovSolver
   opts = {'ksp_type': 'cg',
           'ksp_monitor': None,
           'pc_type': 'gamg'}
   solver = KrylovSolver(a,fes, solverParameters=opts)
   gfu = GridFunction(fes, name="gfu")
   solver.solve(f.vec, gfu.vec)
   Draw (gfu)

.. list-table:: Preconditioner performance for PETSc GAMG. 
   :widths: auto
   :header-rows: 1

   * - Preconditioner
     - Iterations
   * - PETSc GAMG
     - 19 (6.27e-08)

To improve the performance of `PETSc GAMG` we pass it the near-null space composed of the rigid body motions.
Using the :code:`near` flag we tell `PETSc KSP` to pass the nullspace as a near nullspace to `PETSc GAMG`. ::

   from ngsPETSc import NullSpace
   rbms = []
   for val in [(1,0), (0,1), (-y,x)]:
      rbm = GridFunction(fes)
      rbm.Set(CF(val))
      rbms.append(rbm.vec)
   nullspace = NullSpace(fes, rbms, near=True)
   opts = {'ksp_type': 'cg',
           'pc_monitor': None,
           'pc_type': 'gamg'}
   solver = KrylovSolver(a,fes, solverParameters=opts, nullspace=nullspace)
   gfu = GridFunction(fes, name="gfu_near")
   solver.solve(f.vec, gfu.vec)
   Draw (gfu)

.. list-table:: Preconditioner performance for PETSc GAMG. 
   :widths: auto
   :header-rows: 1

   * - Preconditioner
     - Iterations
   * - PETSc GAMG
     - 19 (6.27e-08)
   * - PETSc GAMG (with near nullspace)
     - 6 (4.55e-07)
