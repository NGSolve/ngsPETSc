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
   nu = 1e-4; gamma = 1e8
   nu = Parameter(nu)
   gamma = Parameter(gamma)
   b = CoefficientFunction((4*(2*y-1)*(1-x)*x, -4*(2*x-1)*(1-y)*y)) 
   uin = CoefficientFunction((1,0))

In particular, we will consider a high-order Hood-Taylor discretisation of the problem. Such a discretisation can easily be constructed using NGSolve as follows: ::

   from netgen.occ import *
   import netgen.gui
   import netgen.meshing as ngm
   from ngsolve.meshes import MakeStructured2DMesh
   from mpi4py.MPI import COMM_WORLD

   if COMM_WORLD.rank == 0:
      ngmesh = MakeStructured2DMesh(False, 40,40).ngmesh
      mesh = Mesh(ngmesh.Distribute(COMM_WORLD))
   else:
      mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
   V = H1(mesh, order=4, hoprolongation=True, dirichlet=".*")
   V = V**mesh.dim
   Q = L2(mesh, order=3)
   u,v = V.TnT(); p,q = Q.TnT()
   a = BilinearForm(nu*InnerProduct(Grad(u),Grad(v))*dx)
   a += InnerProduct(Grad(u)*b,v)*dx
   a += gamma*div(u)*div(v)*dx
   b = BilinearForm(div(u)*q*dx)
   c = BilinearForm(-1e-10*p*q*dx)
   f = LinearForm(V)
   g = LinearForm(Q)

In :doc:`stokes.py` we have seen that augmenting the Stokes linear system with a :math:`\gamma(div(\vec{u}),div(\vec{v}))_{L^2(\Omega)}` we can obtain better converge result for a field-split preconditioner where we use the pressure mass matrix instead of the Schur complement. ::

   mG = BilinearForm((1/nu+gamma)*p*q*dx)

We can now assemble the matrices and the right-hand side of the problem. ::

   from ngsPETSc.pc import * 
   a.Assemble()
   aCoarsePre = PETScPreconditioner(a.mat, V.FreeDofs(),
                                    solverParameters={"pc_type":"lu"})
   for l in range(2):
      if l != 0: mesh.Refine()
   V.Update(); Q.Update()
   dofs = BitArray(V.ndof+Q.ndof); dofs[:] = True
   a.Assemble(); b.Assemble(); mG.Assemble();
   f.Assemble(); g.Assemble(); c.Assemble();
   prol = V.Prolongation().Operator(1)
   mGpre = PETScPreconditioner(mG.mat, Q.FreeDofs(),
                                       solverParameters={"pc_type":"lu"})
   K = BlockMatrix( [ [a.mat, b.mat.T], [b.mat, c.mat] ] )

As discussed in :doc:`stokes.py`, the hard part remains the construction of :math:`(A+\gamma B^TM^{-1}B)^{-1}` to precondition the (1,1) block.
We will use a two-level additive Schwarz preconditioner made of an exact coarse correction and a vertex patch smoother, similar to what we presented in :doc:`poisson.py`.
Notice that while the smoother is very similar to the one used in :doc:`poisson.py`, for the coarse correction we are here using h-multigrid and not p-multigrid. ::

   def VertexStarPatchBlocks(mesh, fes):
      blocks = []
      freedofs = fes.FreeDofs()
      for v in mesh.vertices:
         vdofs = set(d for d in fes.GetDofNrs(v) if freedofs[d])
         for ed in mesh[v].edges:
            vdofs |= set(d for d in fes.GetDofNrs(ed) if freedofs[d])
         blocks.append(vdofs)
      return blocks

   blocks = VertexStarPatchBlocks(mesh, V)
   dofs = BitArray(V.ndof); dofs[:] = True
   smoother = ASMPreconditioner(a.mat, dofs, blocks=blocks,
                                solverParameters={"pc_type": "asm",
                                                  "sub_ksp_type": "preonly",
                                                  "sub_pc_type": "lu"})
   prol = V.Prolongation().Operator(1)
   two_lv = prol@ aCoarsePre @ prol.T + smoother
   C = BlockMatrix( [ [two_lv, None], [None, mGpre] ] )
   print("-----------|Additive h-Multigird + Vertex star smoothing|-----------")
   gfu = GridFunction(V, name='AdditiveVel'); gfp = GridFunction(Q, name='AdditivePres')
   gfu.vec.data[:] = 0; gfp.vec.data[:] = 0
   gfu.Set(uin, definedon=mesh.Boundaries("top"))
   rhs = BlockVector( [f.vec, g.vec] )   
   sol = BlockVector( [gfu.vec, gfp.vec] )
   rhs -= K * sol
   dofs = BitArray(V.ndof+Q.ndof); dofs[:] = True
   solver = KrylovSolver(K,dofs, p=C,
                         solverParameters={"ksp_type": "gmres",
                                           "ksp_max_it":100,
                                           "ksp_monitor_true_residual": None,
                                           "ksp_rtol": 1e-6,
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
   vtk = VTKOutput(ma=mesh, coefs=[gfu],
                names = ["velocity"],
                filename="output/Oseen_{}".format(nu.Get()),
                subdivision=0)
   vtk.Do()


.. list-table:: Preconditioner performance for different values of the Reynolds number, for a fixed penalty parameter :math:`\gamma=10^8`
   :widths: auto
   :header-rows: 1

   * - Raynolds number
     - 1e-2
     - 1e-3
     - 1e-4
   * - Augmented Lagrangian preconditioner
     - 2 (1.57e-5)
     - 3 (7.44e-6)
     - 5 (6.59e-6)

