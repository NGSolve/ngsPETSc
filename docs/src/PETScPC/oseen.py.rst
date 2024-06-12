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
   gamma = Parameter(1e6)
   b = CoefficientFunction((4*(2*y-1)*(1-x)*x, -4*(2*x-1)*(1-y)*y)) 
   #b = CoefficientFunction((0,0)) 

In particular, we will consider a high-order Hood-Taylor discretization of the problem. Such a discretization can easily be constructed using NGSolve as follows: ::

   from netgen.occ import *
   import netgen.gui
   import netgen.meshing as ngm
   from mpi4py.MPI import COMM_WORLD

   if COMM_WORLD.rank == 0:
      shape = Rectangle(1,1).Face()
      shape.edges.Max(Y).name="top"
      geo = OCCGeometry(shape, dim=2)
      ngmesh = geo.GenerateMesh(maxh=0.3)
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
We first invert this block using a direct LU factorisation so that we can see what solution we are aiming for. ::

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
   mGpre = PETScPreconditioner(mG.mat, Q.FreeDofs(),
                                       solverParameters={"pc_type":"lu"})
   apre = PETScPreconditioner(a.mat, V.FreeDofs(),
                              solverParameters={"pc_type":"lu"})
   K = BlockMatrix( [ [a.mat, b.mat.T], [b.mat, None] ] )
   C = BlockMatrix( [ [apre, None], [None, mGpre] ] )
   
   uin = CoefficientFunction( (1, 0) )
   luGfu = GridFunction(V, name="LUVel"); luGfp = GridFunction(Q, name="LUPres")
   luGfu.Set(uin, definedon=mesh.Boundaries("top"))
   sol = BlockVector( [luGfu.vec, luGfp.vec] )
   rhs = BlockVector( [f.vec, g.vec] )

   print("-----------|LU|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-10,
                   maxsteps=10, printrates=True, initialize=False)
   Draw(luGfu)

.. list-table:: Preconditioners performance
   :widths: auto
   :header-rows: 1

   * - Preconditioner
     - Iterations
   * - LU
     - 2 (4.07e-8)

To overcome this issue of inverting the agumented (1,1) block, we will construct a two-level additive Schwarz preconditioner made of an exact coarse correction and a vertex patch smoother, similar to what we have done in :doc:`stokes.py`_.
Notice that while the smoother is very similar to the one used in :doc:`stokes.py`, for the coarse correction we are here using h-multigrid and not p-multigrid. ::

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
   gfu = GridFunction(V, name="MG"); gfp = GridFunction(Q)
   gfu.vec.data[:] = 0; gfp.vec.data[:] = 0
   gfu.Set(uin, definedon=mesh.Boundaries("top"))
   sol = BlockVector( [gfu.vec, gfp.vec] )
   rhs = BlockVector( [f.vec, g.vec] )

   print("-----------|Additive h-Multigird + Vertex star smoothing|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-10,
                   maxsteps=10, printrates=True, initialize=False)

.. list-table:: Preconditioners performance
   :widths: auto
   :header-rows: 1

   * - Preconditioner
     - Iterations
   * - LU
     - 2 (4.07e-8)
   * - Additive h-Multigird + Vertex star smoothing
     - 100 (2.08)  

The two-level additive Schwarz preconditioner doesn't seem to be very effective.
For this reason, we decided to opt for a multiplicative multigrid preconditioner where we smoothing step is conducted using NGSolve's own :code:`GMRes`. ::

   class MGPreconditioner(BaseMatrix):
      def __init__ (self, fes, a, coarsepre, smoother):
         super().__init__()
         self.fes = fes
         self.a = a
         self.coarsepre = coarsepre
         self.smoother = smoother
      
      def prol(self, lv):
         return self.fes.Prolongation().Operator(lv)

      def Mult (self, d, w):
         smoother.setActingDofs(self.fes.FreeDofs())
         w[:] = 0
         w += solvers.GMRes(self.a.mat, d, pre=smoother, x=w, maxsteps = 10, printrates=False)
         r = d.CreateVector()
         r.data = d - self.a.mat * w
         w += self.prol(1) @ self.coarsepre @ self.prol(1).T * r
         r.data = d - self.a.mat * w
         #w += smoother * (self.a.mat * w-d)

      def Shape (self):
            return self.mat.shape
      def CreateVector (self, col):
            return self.a.mat.CreateVector(col)

   ml_pre = MGPreconditioner(V, a, aCoarsePre, smoother)
   S = BlockMatrix( [ [IdentityMatrix(V.ndof), -ml_pre@b.mat.T], [None, IdentityMatrix(Q.ndof)]] )
   ST = BlockMatrix( [ [IdentityMatrix(V.ndof), None], [-b.mat@ml_pre, IdentityMatrix(Q.ndof)]] )
   C = S@BlockMatrix( [ [ml_pre, None], [None, mGpre] ] )@ST
   ngsGfu = GridFunction(V, name="ngs"); ngsGfp = GridFunction(Q)
   ngsGfu.vec.data[:] = 0; ngsGfp.vec.data[:] = 0
   ngsGfu.Set(uin, definedon=mesh.Boundaries("top"))
   sol = BlockVector( [ngsGfu.vec, ngsGfp.vec] )
   rhs = BlockVector( [f.vec, g.vec] )

   print("-----------|NGS MinRES Multiplicative h-Multigird + Vertex star GMRES relaxetion|-----------")
   solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, tol=1e-10,
                   maxsteps=10, printrates=True, initialize=False)
   Draw(ngsGfu)

.. list-table:: Preconditioners performance
   :widths: auto
   :header-rows: 1

   * - Preconditioner
     - Iterations
   * - LU
     - 2 (4.07e-8)
   * - Additive h-Multigird + Vertex star smoothing
     - 100 (2.08)  
   * - Multiplicative h-Multigird + Vertex star smoothing
     - 100 (0.09)  

::

   print("-----------|PETSc Multiplicative h-Multigird + Vertex star GMRES relaxetion|-----------")
   dofs = BitArray(V.ndof+Q.ndof); dofs[:] = True
   gfu = GridFunction(V, name='PETScVel'); gfp = GridFunction(Q, name='PETScPres')
   gfu.vec.data[:] = 0; gfp.vec.data[:] = 0
   gfu.Set(uin, definedon=mesh.Boundaries("top"))
   rhs = BlockVector( [f.vec, g.vec] )   
   sol = BlockVector( [gfu.vec, gfp.vec] )
   rhs -= K * sol
 
   solver = KrylovSolver(K,dofs, p=C,
                         solverParameters={"ksp_type": "lgmres",
                                           "ksp_max_it":100,
                                           "ksp_rtol": 1e-14,
                                           #"ksp_monitor":  None,
                                           "ksp_monitor_true_residual": None,
                                           "pc_type": "mat"
                                           })
   solver.solve(rhs, sol)
   gfu0 = GridFunction(V, name="PETSc0"); gfp0 = GridFunction(Q)
   gfu0.vec.data[:]= 0
   gfu0.Set(uin, definedon=mesh.Boundaries("top"))
   sol0 = BlockVector( [gfu0.vec, gfp0.vec] )
   sol += sol0
   gfu.vec.data = sol[0]
   Draw(gfu)