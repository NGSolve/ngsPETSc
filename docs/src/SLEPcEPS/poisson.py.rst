Solving the Laplace eigenvalue problem using SLEPc EPS
=======================================================

In this tutorial, we explore using `SLEPc EPS` to solve the Laplace eigenvalue problem in primal and mixed formulations.
We begin by considering the Poisson eigenvalue problem in primal form, i.e.

.. math::

   \text{find } (u,\lambda) \in H^1_0(\Omega)\times\mathbb{R} \text{ s.t. } a(u,v) := \int_{\Omega} \nabla u\cdot \nabla v \; d\vec{x} = \lambda (u,v)_{L^2(\Omega)},\quad v\in H^1_0(\Omega).

Such a discretisation can easily be constructed using NGSolve as follows: ::

   from ngsolve import *
   import netgen.gui
   import netgen.meshing as ngm
   import numpy as np
   from mpi4py.MPI import COMM_WORLD

   mesh = Mesh(unit_square.GenerateMesh(maxh=0.1, comm=COMM_WORLD))

   order = 3
   fes = H1(mesh, order=order, dirichlet="left|right|top|bottom")
   print("Number of degrees of freedom: ", fes.ndof)
   u,v = fes.TnT()
   a = BilinearForm(grad(u)*grad(v)*dx, symmetric=True)
   a.Assemble()
   m = BilinearForm(-1*u*v*dx, symmetric=True)
   m.Assemble()

We then proceed to solve the eigenvalue problem using `SLEPc EPS` which is wrapped by ngsPETSc's :code:`EigenSolver` class.
In particular, we will use the SLEPc implementation of locally optimal block preconditioned conjugate gradient.
Notice that we have assembled the mass matrix so that it is symmetric negative definite; this is because ngsPETSc :code:`Eigensolver` requires all eigenvalue problems to be written as a polynomial eigenvalue problem, i.e.

.. math::
   A\vec{U} - \lambda M\vec{U} = 0

where :math:`A` and :math:`M` are the stiffness and mass matrices, respectively. ::

   from ngsPETSc import EigenSolver
   solver = EigenSolver((m, a), fes, 10, solverParameters={"eps_type":"lobpcg", "st_type":"precond"})
   solver.solve()
   for i in range(10):
      print("Normalised (by pi^2) Eigenvalue ", i, " = ", solver.eigenValue(i)/(np.pi*np.pi))
   modes, _ = solver.eigenFunctions(list(range(10)))
   Draw(modes)

Notice that we can access a single eigenvalue using :code:`solver.eigenValue(i)` and the corresponding eigenfunction using :code:`solver.eigenFunction(i)`.
If we use :code:`solver.eigenFunctions(indices)` we can access the first 10 eigenfunctions as a multi-dimensional array and we can obtain the corresponding eigenvalues using :code:`solver.eigenValues(indices)`.
We now consider a mixed formulation of the Laplace eigenvalue problem, i.e.

.. math::

   \text{find } (\vec{\sigma}, u, \lambda) \in H(\text{div},\Omega)\times L^2(\Omega)\times \mathbb{R} \text{ s.t. } \\
   \begin{cases}
      \int_{\Omega} \vec{\sigma}\cdot\vec{\tau} \; d\vec{x} - \int_{\Omega} u \nabla \cdot \vec{\tau} \; d\vec{x} = 0 & \forall \vec{\tau}\in H(\text{div},\Omega)\\
      \int_{\Omega} \nabla\cdot\vec{\sigma}v \; d\vec{x} = \lambda \int_{\Omega} uv \; d\vec{x} & \forall v\in L^2(\Omega)
   \end{cases}

We can discretise this problem using NGSolve as follows: ::

   V = HDiv(mesh, order=order)
   Q = L2(mesh, order=order-1)
   W = FESpace([V,Q])
   print("Number of degrees of freedom: ", W.ndof)
   sigma, u = W.TrialFunction()
   tau, v = W.TestFunction()
   a = BilinearForm(sigma*tau*dx + div(tau)*u*dx + div(sigma)*v*dx)
   a.Assemble()

   m = BilinearForm(1*u*v*dx)
   m.Assemble()

We can again solve the eigenvalue problem using ngsPETSc's `EigenSolver` class.
The mass matrix now has a large kernel, hence is no longer symmetric positive definite, therefore we can not use LOBPCG as a solver.
Instead, we will use a Krylov-Schur solver with a shift-and-invert spectral transformation to target the smallest eigenvalues.
Notice that because we are using a shift-and-invert spectral transformation we only need to invert the stiffness matrix which has a trivial kernel since we are using an inf-sup stable discretisation.
If we tried to use a simple shift transformation to target the largest eigenvalues we would have run into the error of trying to invert a singular matrix.::
   
   solver = EigenSolver((m, a), W, 10,
                        solverParameters={"eps_type":"krylovschur", 
                                          "st_type":"sinvert",
                                          "eps_target": 0,
                                          "st_pc_type": "lu",
                                          "st_pc_factor_mat_solver_type": "mumps"})
   solver.solve()
   for i in range(10):
      print("Normalised (by pi^2) Eigenvalue ", i, " = ", solver.eigenValue(i)/(np.pi*np.pi))
   mode, _ = solver.eigenFunction(0)
   modeSigma, modeU = mode.components
   Draw(modeSigma, mesh, "sigma")
   Draw(modeU, mesh, "u")
