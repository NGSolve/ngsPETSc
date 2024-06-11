Vertex Patch smoothing for Augmented Lagrangian formulations of the Oseen problem
===================================================================================

In this tutorial, we will see how to use an augmented Lagrangian formulation to precondition the Oseen problem, i.e.

.. math::

   \text{Given } \vec{\beta} \in \mathbb{R}^3 \text{ find } (\vec{u}, p) \in [H^1_{0}(\Omega)]^d \times L^2(\Omega) \text{ s.t. }

   \begin{cases} 
      \nu (\nabla \vec{u}, \nabla \vec{v})_{L^2(\Omega)} + (\nabla \cdot \vec{v}, p)_{L^2(\Omega)} - (\nabla \vec{u} \vec{\beta}, \vec{v})_{L^2(\Omega)} + \gamma (\text{div}(\vec{u}), \text{div}(\vec{v}))_{L^2(\Omega)} = (\vec{f}, \vec{v})_{L^2(\Omega)} \quad \forall v \in H^1_{0}(\Omega) \\
      (\nabla \cdot \vec{u}, q)_{L^2(\Omega)} = 0 \quad \forall q \in L^2(\Omega)
   \end{cases}

Let us begin defining the parameters of the problem. ::

   from ngsolve import *
   from ngsPETSc import *
   nu = Parameter(1e-4)
   gamma = Parameter(1e4)
   b = CoefficientFunction((0,0))

In particular, we will consider a high-order Hood-Taylor discretization of the problem. Such a discretization can easily be constructed using NGSolve as follows: ::

   from netgen.occ import *
   import netgen.gui
   import netgen.meshing as ngm
   from mpi4py.MPI import COMM_WORLD

   if COMM_WORLD.rank == 0:
      shape = Rectangle(1,1).Face()
      shape.edges.Max(Y).name="top"
      geo = OCCGeometry(shape, dim=2)
      ngmesh = geo.GenerateMesh(maxh=0.1)
      mesh = Mesh(ngmesh.Distribute(COMM_WORLD))
   else:
      mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
   V = H1(mesh, order=4, hoprolongation=True, dirichlet=".*")
   V = V**mesh.dim
   Q = H1(mesh, order=3)
   u,v = V.TnT(); p,q = Q.TnT()
   a = BilinearForm(nu*InnerProduct(Grad(u),Grad(v))*dx)
   a += InnerProduct(Grad(u)*b,v)*dx
   a += gamma*div(u)*div(v)*dx
   b = BilinearForm(div(u)*q*dx)
   f = LinearForm(V)
   g = LinearForm(Q)

In :doc:`stokes.py` we have seen that augmenting the Stokes linear system with a :math:`\gamma(div(\vec{u}),div(\vec{v}))_{L^2(\Omega)}` we can obtain better converge result for a filed split preconditioner where we use the pressure mass matrix instead of the Schur complement.
Let us then construct the augmentation block of the matrix and the pressure mass matrix to use in the (2,2) block of our preconditioner. ::

   mG = BilinearForm((1/nu+gamma)*p*q*dx)

As discussed in :doc:`stokes.py`, the hard part remains the construction of the :math:`(A+\gamma B^TM^{-1}B)^{-1}` to precondition the (1,1) block.
We begin observing that a classical h-multigird preconditioner is not suitable for this problem, due to the large kernel introduced by the augmentation block.
We construct a classical h-multigird using the `hoprolongation` flag when constructing the finite element space. ::

   from ngsPETSc.pc import * 
   a.Assemble()
   aCoarsePre = PETScPreconditioner(a.mat, V.FreeDofs(),
                                    solverParameters={"pc_type":"lu"})
   for l in range(2):
      if l != 0: mesh.Refine()
   V.Update(); Q.Update()
   a.Assemble(); b.Assemble(); mG.Assemble();
   f.Assemble(); g.Assemble();
   prol = V.Prolongation().Operator(1)
   smoother = PETScPreconditioner(a.mat, V.FreeDofs(),
                                  solverParameters={"pc_type": "jacobi"})
   apre = prol @ aCoarsePre @ prol.T + smoother
   mGpre = PETScPreconditioner(mG.mat, Q.FreeDofs(),
                                       solverParameters={"pc_type":"lu"})

   K = BlockMatrix( [ [a.mat, b.mat.T], [b.mat, None] ] )
   C = BlockMatrix( [ [apre, None], [None, mGpre] ] )
   
   uin = CoefficientFunction( (1, 0) )
   gfu = GridFunction(V); gfp = GridFunction(Q)
   gfu.Set(uin, definedon=mesh.Boundaries("top"))
   sol = BlockVector( [gfu.vec, gfp.vec] )
   rhs = BlockVector( [f.vec, g.vec] )

   print("-----------|Additive h-Multigird|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-10,
                   maxsteps=100, printrates=True, initialize=False)
   Draw(gfu)


To overcome this issue we will construct a two-level additive Schwarz preconditioner made of an exact coarse correction and a vertex patch smoother.
Notice that while the smoother is very similar to the one used in :doc:`poisson.py`, for the coarse correction we are here using h-multigrid and not p-multigrid. ::

   def VertexStarPatchBlocks(mesh, fes):
      blocks = []
      freedofs = fes.FreeDofs()
      for v in mesh.vertices:
         vdofs = set(d for d in fes.GetDofNrs(v) if freedofs[d])
         for ed in mesh[v].edges:
            vdofs |= set(d for d in fes.GetDofNrs(ed) if freedofs[d])
         for fc in mesh[v].faces:
            vdofs |= set(d for d in fes.GetDofNrs(fc) if freedofs[d])
         blocks.append(vdofs)
      return blocks

   blocks = VertexStarPatchBlocks(mesh, V)
   dofs = BitArray(V.ndof); dofs[:] = True
   smoother = ASMPreconditioner(a.mat, dofs, blocks=blocks,
                                solverParameters={"pc_type": "asm",
                                                  "sub_ksp_type": "preonly",
                                                  "sub_pc_type": "lu"})
   two_lv = apre + smoother
   C = BlockMatrix( [ [two_lv, None], [None, mGpre] ] )
   
   gfu.vec.data[:] = 0; gfp.vec.data[:] = 0
   gfu.Set(uin, definedon=mesh.Boundaries("top"))
   sol = BlockVector( [gfu.vec, gfp.vec] )
   rhs = BlockVector( [f.vec, g.vec] )

   print("-----------|Additive h-Multigird + Vertex star relaxetion|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-10,
                   maxsteps=100, printrates=True, initialize=False)
   Draw(gfu)

We try a multiplicative preconditioner instead ::

   from ngsPETSc import KrylovSolver
   class MGPreconditioner(BaseMatrix):
      def __init__ (self, fes, a, coarsepre, smoother):
         super().__init__()
         self.fes = fes
         self.a = a
         self.coarsepre = coarsepre
         self.smoother = smoother
         self.prol = fes.Prolongation().Operator(1)

      def Mult (self, d, w):
         prj = Projector(mask=self.fes.FreeDofs(), range=True) 
         smoother = prj @ self.smoother @ prj.T
         w[:] = 0
         w += smoother*(d-self.a.mat*w)
         r = d.CreateVector()
         r.data = d - self.a.mat * w
         w += self.prol @ self.coarsepre @ self.prol.T * r

      def Shape (self):
            return self.mat.shape
      def CreateVector (self, col):
            return self.mat.CreateVector(col)

   ml_pre = MGPreconditioner(V, a, aCoarsePre, smoother)
   C = BlockMatrix( [ [ml_pre, None], [None, mGpre] ] )
   gfu.vec.data[:] = 0; gfp.vec.data[:] = 0
   gfu.Set(uin, definedon=mesh.Boundaries("top"))
   sol = BlockVector( [gfu.vec, gfp.vec] )
   rhs = BlockVector( [f.vec, g.vec] )

   print("-----------|Multiplicative h-Multigird + Vertex star GMRES relaxetion|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-10,
                   maxsteps=100, printrates=True, initialize=False)
   Draw(gfu)