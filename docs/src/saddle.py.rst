Saddle point problems and PETSc PC
=======================================

In this tutorial we explore solving constructing preconditioners for saddle point problems using `PETSc PC`.
We begin by creating a discretisation of the Poisson problem using H1 elements, in particular we consider the usual variational formulation
.. math::

   \text{find } u\in H^1_{0,0}(\Omega) \text{ s.t. } a(u,v) := \int_{\Omega} \nabla u\cdot \nabla v \; d\vec{x} = L(v) := \int_{\Omega} fv\; d\vec{x}\qquad v\in H^1_{0,0}(\Omega).

Such a discretisation can easily be constructed using NGSolve as follows: ::

   from ngsolve import *
   from ngsolve import BilinearForm as BF
   from netgen.occ import *
   import netgen.gui
   import netgen.meshing as ngm
   from mpi4py.MPI import COMM_WORLD

   if COMM_WORLD.rank == 0:
      shape = Rectangle(2,0.41).Circle(0.2,0.2,0.05).Reverse().Face()
      shape.edges.name="wall"
      shape.edges.Min(X).name="inlet"
      shape.edges.Max(X).name="outlet"
      geo = OCCGeometry(shape, dim=2)
      ngmesh = geo.GenerateMesh(maxh=0.05)
      mesh = Mesh(ngmesh.Distribute(COMM_WORLD))
   else:
      mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
   V = VectorH1(mesh, order=2, dirichlet="wall|inlet|cyl")
   Q = H1(mesh, order=1)
   u,v = V.TnT(); p,q = Q.TnT()
   a = BilinearForm(InnerProduct(Grad(u),Grad(v))*dx)
   a.Assemble()
   b = BilinearForm(div(u)*q*dx)
   b.Assemble()
   gfu = GridFunction(V, name="u")
   gfp = GridFunction(Q, name="p")
   uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
   gfu.Set(uin, definedon=mesh.Boundaries("inlet"))
   f = LinearForm(V).Assemble()
   g = LinearForm(Q).Assemble();

We can now explore what happens if we use a Schour complement preconditioner for the saddle point problem.
We can construct the Schur complement preconditioner using the following code: ::

   K = BlockMatrix( [ [a.mat, b.mat.T], [b.mat, None] ] )
   S = (b.mat @ a.mat.Inverse(V.FreeDofs()) @ b.mat.T).ToDense().NumPy()
   from numpy.linalg import inv
   from scipy.sparse import coo_matrix
   from ngsolve.la import SparseMatrixd 
   Sinv = coo_matrix(inv(S))
   Sinv = la.SparseMatrixd.CreateFromCOO(indi=Sinv.row, indj=Sinv.col, values=Sinv.data, h=S.shape[0], w=S.shape[1])
   C = BlockMatrix( [ [a.mat.Inverse(V.FreeDofs()), None], [None, Sinv] ] )

   rhs = BlockVector ( [f.vec, g.vec] )
   sol = BlockVector( [gfu.vec, gfp.vec] )

   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, printrates=True, initialize=False);
   Draw(gfu)