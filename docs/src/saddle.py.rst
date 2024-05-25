Saddle point problems and PETSc PC
=======================================

In this tutorial we explore solving constructing preconditioners for saddle point problems using `PETSc PC`.
We begin by creating a discretisation of the Poisson problem using H1 elements, in particular we consider the usual variational formulation

.. math::       
   
   \text{find } (vec{u},p) \in [H^1_{0}(\Omega)]^d\times L^2(\Omega) \text{ s.t. }
   
   \begin{cases} 
      (\nabla \vec{u},\nabla \vec{v})_{L^2(\Omega)} + (\nabla\cdot \vec{v}, p)_{L^2(\Omega)}  = (\vec{f},\vec{v})_{L^2(\Omega)} \qquad v\in H^1_{0}(\Omega)\\
      (\nabla\cdot \vec{u},q)_{L^2(\Omega)} = 0 \qquad q\in L^2(\Omega)
   \end{cases}

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
      ngmesh = geo.GenerateMesh(maxh=0.1)
      mesh = Mesh(ngmesh.Distribute(COMM_WORLD))
   else:
      mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
   nu = Parameter(1.0)
   V = VectorH1(mesh, order=2, dirichlet="wall|inlet|cyl")
   Q = L2(mesh, order=0)
   u,v = V.TnT(); p,q = Q.TnT()
   a = BilinearForm(nu*InnerProduct(Grad(u),Grad(v))*dx)
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
   from ngsPETSc import pc
   apre = Preconditioner(a, "PETScPC", pc_type="lu")
   S = (b.mat @ apre.mat @ b.mat.T).ToDense().NumPy()
   from numpy.linalg import inv
   from scipy.sparse import coo_matrix
   from ngsolve.la import SparseMatrixd 
   Sinv = coo_matrix(inv(S))
   Sinv = la.SparseMatrixd.CreateFromCOO(indi=Sinv.row, 
                                         indj=Sinv.col,
                                         values=Sinv.data,
                                         h=S.shape[0],
                                         w=S.shape[1])
   C = BlockMatrix( [ [a.mat.Inverse(V.FreeDofs()), None],
                      [None, Sinv] ] )

   rhs = BlockVector (  [f.vec, g.vec] )
   sol = BlockVector( [gfu.vec, gfp.vec] )
   print("-----------|Schur|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-8,
                   printrates=True, initialize=False)
   Draw(gfu)

Notice that the Schur complement is dense hence inverting it is not a good idea. Not only that but to perform the inversion of the Schur complement had to write a lot of "boiler plate" code.
Since our discretisation is inf-sup stable it is possible to prove that the mass matrix of the pressure space is spectrally equivalent to the Schur complement.
This means that we can use the mass matrix of the pressure space as a preconditioner for the Schur complement.
Notice that we still need to invert tha mass matrix and we will do so using a `PETSc PC` of type Jacobi, which is the exact inverse since we are using `P0` elements.
We will also invert the Laplacian block using a `PETSc PC` of type `LU`. ::

   m = BilinearForm((1/nu)*p*q*dx).Assemble()
   mpre = Preconditioner(m, "PETScPC", pc_type="jacobi")
   apre = Preconditioner(a, "PETScPC", pc_type="lu")
   C = BlockMatrix( [ [apre.mat, None], [None, mpre.mat] ] )

   gfu.vec.data[:] = 1; gfp.vec.data[:] = 1
   sol = BlockVector( [gfu.vec, gfp.vec] )

   print("-----------|Mass LU|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-8,
                   printrates=True, initialize=False)
   Draw(gfu)

The mass matrix as a preconditioner doesn't seem to be ideal, in fact our Krylov solver took many iterations to converge.
To resolve this issue we resort to an augmented Lagrangian formulation, i.e.

.. math::
   \begin{cases} 
      (\nabla \vec{u},\nabla \vec{v})_{L^2(\Omega)} + (\nabla\cdot \vec{v}, p)_{L^2(\Omega)} + \gamma (\nabla\cdot \vec{u},\nabla\cdot\vec{v})_{L^2(\Omega)} = (\vec{f},\vec{v})_{L^2(\Omega)} \qquad v\in H^1_{0}(\Omega)\\
      (\nabla\cdot \vec{u},q)_{L^2(\Omega)} = 0 \qquad q\in L^2(\Omega)
   \end{cases}

This formulation can easily be adding an augmentation block in the `BlockMatrix`, as follows: ::

   gamma = Parameter(1e6)
   aG = BilinearForm(nu*InnerProduct(Grad(u),Grad(v))*dx+gamma*div(u)*div(v)*dx)
   aG.Assemble()
   aGpre = Preconditioner(aG, "PETScPC", pc_type="lu")
   mG = BilinearForm((1/nu+gamma)*p*q*dx).Assemble()
   mGpre = Preconditioner(mG, "PETScPC", pc_type="jacobi")
   
   K = BlockMatrix( [ [aG.mat, b.mat.T], [b.mat, None] ] )
   C = BlockMatrix( [ [aGpre.mat, None], [None, mGpre.mat] ] )

   gfu.vec.data[:] = 1; gfp.vec.data[:] = 1
   sol = BlockVector( [gfu.vec, gfp.vec] )

   print("-----------|Augmented LU|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-11,
                   printrates=True, initialize=False)
   Draw(gfu)

Notice that so far we have been inverting the matrix corresponding to the Laplacian block using a direct LU factorisation.
As our mesh becomes finer and finer this is no longer a viable options. To overcome this issue we can try inverting the matrix via `HYPRE`. ::

   aGpre = Preconditioner(aG, "PETScPC", pc_type="hypre")
   C = BlockMatrix( [ [aGpre.mat, None], [None, mGpre.mat] ] )
   gfu.vec.data[:] = 1; gfp.vec.data[:] = 1
   sol = BlockVector( [gfu.vec, gfp.vec] )

   print("-----------|Augmented HYPRE|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-10,
                   printrates=True, initialize=False)
   Draw(gfu)

We notice that our solver is no longer converging. This a known issue of augemnted Lagrangian formulation: inverting the augmented Laplacian block using multigrid is hard.
The reason behind this phenomena is the fact that the augmented Laplacian block has a large kernel. Lets try to fix this using a vertex patch two level additive Schwartz preconditioner, which is known to be kernel capturing. ::

   ngmesh = unit_square.GenerateMesh(maxh=0.1)
   mesh = Mesh(ngmesh)
   V = H1(mesh, order=2)
   u,v = V.TnT()
   aG = BilinearForm(InnerProduct(grad(u),grad(v))*dx)
   aG.Assemble()
   print(aG.mat.shape)
   for l in range(3):
      mesh.ngmesh.Refine(adaptive=True)
      V.Update()
      aG.Assemble()
      prol = V.Prolongation().Operator(l+1)
      print(prol.shape, aG.mat.shape)