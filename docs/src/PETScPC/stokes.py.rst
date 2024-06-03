Saddle point problems and PETSc PC
=======================================

In this tutorial, we explore constructing preconditioners for saddle point problems using `PETSc PC`.
In particular, we will consider a Bernardi-Raugel inf-sup stable discretization of the Stokes problem, i.e.

.. math::       
   
   \text{find } (\vec{u},p) \in [H^1_{0}(\Omega)]^d\times L^2(\Omega) \text{ s.t. }
   
   \begin{cases} 
      (\nabla \vec{u},\nabla \vec{v})_{L^2(\Omega)} + (\nabla\cdot \vec{v}, p)_{L^2(\Omega)}  = (\vec{f},\vec{v})_{L^2(\Omega)} \qquad v\in H^1_{0}(\Omega)\\
      (\nabla\cdot \vec{u},q)_{L^2(\Omega)} = 0 \qquad q\in L^2(\Omega)
   \end{cases}

Such a discretization can easily be constructed using NGSolve as follows: ::

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
   V = VectorH1(mesh, order=4, dirichlet="wall|inlet|cyl", autoupdate=True)
   Q = L2(mesh, order=2, autoupdate=True)
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
   from ngsPETSc.pc import *
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

Notice that the Schur complement is dense hence inverting it is not a good idea. Not only that but to perform the inversion of the Schur complement had to write a lot of "boilerplate" code.
Since our discretization is inf-sup stable it is possible to prove that the mass matrix of the pressure space is spectrally equivalent to the Schur complement.
This means that we can use the mass matrix of the pressure space as a preconditioner for the Schur complement.
Notice that we still need to invert the mass matrix and we will do so using a `PETSc PC` of type Jacobi, which is the exact inverse since we are using `P0` elements.
We will also invert the Laplacian block using a `PETSc PC` of type `LU`. ::

   m = BilinearForm((1/nu)*p*q*dx).Assemble()
   mpre = Preconditioner(m, "PETScPC", pc_type="lu")
   apre = a.mat.Inverse(V.FreeDofs())
   C = BlockMatrix( [ [apre, None], [None, mpre] ] )

   gfu.vec.data[:] = 0; gfp.vec.data[:] = 0;
   gfu.Set(uin, definedon=mesh.Boundaries("inlet"))
   sol = BlockVector( [gfu.vec, gfp.vec] )

   print("-----------|Mass & LU|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-8,
                   maxsteps=100, printrates=True, initialize=False)
   Draw(gfu)

We can also construct a multi-grid preconditioner for the top left block of the saddle point problem, as we have seen in :doc:`poisson.py`. ::

   def DoFInfo(mesh, fes):
      blocks = []
      freedofs = fes.FreeDofs()
      vertexdofs = BitArray(fes.ndof)
      vertexdofs[:] = False
      for v in mesh.vertices:
         vdofs = set()
         vdofs |= set(d for d in fes.GetDofNrs(v) if freedofs[d])
         for ed in mesh[v].edges:
            vdofs |= set(d for d in fes.GetDofNrs(ed) if freedofs[d])
         for fc in mesh[v].faces:
            vdofs |= set(d for d in fes.GetDofNrs(fc) if freedofs[d])
         blocks.append(vdofs)
         for d in fes.GetDofNrs(v):
            vertexdofs[d] = True
      vertexdofs &= fes.FreeDofs()
      return vertexdofs, blocks 

   vertexdofs, blocks = DoFInfo(mesh, V)
   blockjac = a.mat.CreateBlockSmoother(blocks)
   preH = PETScPreconditioner(a.mat, vertexdofs, solverParameters={"pc_type":"hypre"})
   twolvpre = preH + blockjac
   C = BlockMatrix( [ [twolvpre, None], [None, mpre] ] )
   gfu.vec.data[:] = 0; gfp.vec.data[:] = 0;
   gfu.Set(uin, definedon=mesh.Boundaries("inlet"))
   print("-----------|Mass & Two Level Additivew Schwarz|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-8,
                   maxsteps=100, printrates=True, initialize=False)
   

The mass matrix as a preconditioner doesn't seem to be ideal, in fact, our Krylov solver took many iterations to converge with a direct LU factorization of the velocity block and did not converge at all with `HYPRE`.
To resolve this issue we resort to an augmented Lagrangian formulation, i.e.

.. math::
   \begin{cases} 
      (\nabla \vec{u},\nabla \vec{v})_{L^2(\Omega)} + (\nabla\cdot \vec{v}, p)_{L^2(\Omega)} + \gamma (\nabla\cdot \vec{u},\nabla\cdot\vec{v})_{L^2(\Omega)} = (\vec{f},\vec{v})_{L^2(\Omega)} \qquad v\in H^1_{0}(\Omega)\\
      (\nabla\cdot \vec{u},q)_{L^2(\Omega)} = 0 \qquad q\in L^2(\Omega)
   \end{cases}

This formulation can easily be constructed by adding a new velocity block in the `BlockMatrix`, as follows: ::

   gamma = Parameter(1e6)
   aG = BilinearForm(nu*InnerProduct(Grad(u),Grad(v))*dx+gamma*div(u)*div(v)*dx)
   aG.Assemble()
   aGpre = Preconditioner(aG, "PETScPC", pc_type="lu")
   mG = BilinearForm((1/nu+gamma)*p*q*dx).Assemble()
   mGpre = Preconditioner(mG, "PETScPC", pc_type="jacobi")
   
   K = BlockMatrix( [ [aG.mat, b.mat.T], [b.mat, None] ] )
   C = BlockMatrix( [ [aGpre.mat, None], [None, mGpre.mat] ] )

   gfu.vec.data[:] = 0; gfp.vec.data[:] = 0;
   gfu.Set(uin, definedon=mesh.Boundaries("inlet"))
   sol = BlockVector( [gfu.vec, gfp.vec] )

   print("-----------|Augmented LU|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-10,
                   printrates=True, initialize=False)
   Draw(gfu)

Notice that so far we have been inverting the matrix corresponding to the Laplacian block using a direct LU factorization.
This is not ideal for large problems, and we can use a `Hypre` preconditioner for the Laplacian block. ::

   smoother = aG.mat.CreateBlockSmoother(blocks)
   preHG = PETScPreconditioner(aG.mat, vertexdofs, solverParameters={"pc_type":"gamg"})
   twolvpre = preHG + smoother
   C = BlockMatrix( [ [twolvpre, None], [None, mGpre] ] )
   gfu.vec.data[:] = 0; gfp.vec.data[:] = 0;
   gfu.Set(uin, definedon=mesh.Boundaries("inlet"))
   print("-----------|Augmented Two Level Additivew Schwarz|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-10,
                   printrates=True, initialize=False)
   Draw(gfu)

Our first attempt at using a `HYPRE` preconditioner for the Laplacian block did not converge. This is because the top left block of the saddle point problem now contains the augmentation term, which has a very large kernel.
It is well known that algebraic multi-grid methods do not work well with indefinite problems, and this is what we are observing here. ::

Let us consider an alternative approach to the augmented Lagrangian formulation. We begin by constructing the augmented Lagrangian formulation in more numerical linear algebra terms, i.e. ::

   d = BilinearForm((1/gamma)*p*q*dx)
   d.Assemble()
   dpre = PETScPreconditioner(d.mat, Q.FreeDofs(), solverParameters={"pc_type":"lu"})
   aG = a.mat + b.mat.T@dpre@b.mat
   aG = coo_matrix(aG.ToDense().NumPy())
   aG = la.SparseMatrixd.CreateFromCOO(indi=aG.row, 
                                         indj=aG.col,
                                         values=aG.data,
                                         h=aG.shape[0],
                                         w=aG.shape[1])
   K = BlockMatrix( [ [aG, b.mat.T], [b.mat, None] ] )
   pre = PETScPreconditioner(aG, V.FreeDofs(), solverParameters={"pc_type":"lu"})
   C = BlockMatrix( [ [pre, None], [None, mGpre.mat] ] )

   gfu.vec.data[:] = 0; gfp.vec.data[:] = 0;
   gfu.Set(uin, definedon=mesh.Boundaries("inlet"))
   sol = BlockVector( [gfu.vec, gfp.vec] )

   print("-----------|Boffi--Lovadina Augmentation LU|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-10,
                   printrates=True, initialize=False)
   Draw(gfu)

We can now think of a more efficient way to invert the matrix corresponding to the augmentation term.
In fact, since we know that the augmentation block has a lower rank than the Laplacian block, we can use the Sherman-Morrisson-Woodbory formula to invert the augmentation block. ::

   SM = (d.mat + b.mat@apre@b.mat.T).ToDense().NumPy()
   SM = coo_matrix(SM)
   SM = la.SparseMatrixd.CreateFromCOO(indi=SM.row, 
                                         indj=SM.col,
                                         values=SM.data,
                                         h=SM.shape[0],
                                         w=SM.shape[1])
   
   SMinv = PETScPreconditioner(SM, Q.FreeDofs(), solverParameters={"pc_type":"lu"})

   C = BlockMatrix( [ [apre - apre@b.mat.T@SMinv@b.mat@apre, None], [None, mGpre.mat] ] )

   gfu.vec.data[:] = 0; gfp.vec.data[:] = 0;
   gfu.Set(uin, definedon=mesh.Boundaries("inlet"))
   sol = BlockVector( [gfu.vec, gfp.vec] )

   print("-----------|Boffi--Lovadina Augmentation Sherman-Morrisson-Woodbory|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-10,
                   printrates=True, initialize=False)
   Draw(gfu)

   C = BlockMatrix( [ [apre + apre@(b.mat.T@mpre.mat@b.mat)@apre, None], [None, mGpre.mat] ] )

   gfu.vec.data[:] = 0; gfp.vec.data[:] = 0;
   gfu.Set(uin, definedon=mesh.Boundaries("inlet"))
   sol = BlockVector( [gfu.vec, gfp.vec] )

   print("-----------|Boffi--Lovadina Augmentation Sherman-Morrisson-Woodbory|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-13,
                   printrates=True, initialize=False)
   Draw(gfu)