Solving linear elasticity with a near nullspace
==============================================

In this tutorial, we explore a linear elasticity discretisation.
As we did in :doc:`poisson.py`, we will solve the linear system originating from the discretization using a `PETSc KSP`.
We begin by creating a discretisation for the weak formulation of linear elasticity with Lame coefficients :math:`\mu` and :math:`\lambda`, i.e. 

.. math::

   \text{find } \vec{u}\in [H^1_0(\Omega)]^d \text{ s.t. } a(u,v) := 2\mu \int_{\Omega} \epsilon(\vec{u}) : \epsilon(\vec{v}) \; d\vec{x} + \lambda \int_\Omega (\nabla \cdot \vec{u})\; d\vec{x} = L(v) := \int_{\Omega} fv\; d\vec{x}\qquad \vec{v}\in [H^1_0(\Omega)]^d.

Such a discretisation can easily be constructed using NGSolve as follows: ::

   from ngsolve import *
   import netgen.gui
   import netgen.meshing as ngm
   from mpi4py.MPI import COMM_WORLD

   if COMM_WORLD.rank == 0:
      mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
   else:
      mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))

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

We begin solving the linear system using `PETSc GAMG` as a preconditioner. ::

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

To improve the performance of `HYPRE` can now construct a null space made of the rigid body motions as follows.
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