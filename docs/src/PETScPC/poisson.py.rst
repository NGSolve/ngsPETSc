Vertex patch smoothing for p-multigrid preconditioners for the Poisson problem
===============================================================================

In this tutorial, we explore using `PETSc PC` as a building block inside the NGSolve preconditioning infrastructure.
Not all the preconditioning strategies are equally effective for the discretisations of the Poisson problem here considered.
This demo is intended to provide a starting point for the exploration of preconditioning strategies rather than providing a definitive answer.
We begin by creating a discretisation of the Poisson problem using H1 elements, in particular, we consider the usual variational formulation

.. math::

   \text{find } u\in H^1_0(\Omega) \text{ s.t. } a(u,v) := \int_{\Omega} \nabla u\cdot \nabla v \; d\vec{x} = L(v) := \int_{\Omega} fv\; d\vec{x}\qquad v\in H^1_0(\Omega).

Such a discretisation can easily be constructed using NGSolve as follows: ::

   from ngsolve import *
   import netgen.gui
   import netgen.meshing as ngm
   from mpi4py.MPI import COMM_WORLD

   if COMM_WORLD.rank == 0:
      ngmesh = unit_square.GenerateMesh(maxh=0.1)
      for _ in range(4):
         ngmesh.Refine()
      mesh = Mesh(ngmesh.Distribute(COMM_WORLD))
   else:
      mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
   order = 2
   fes = H1(mesh, order=order, dirichlet="left|right|top|bottom")
   print("Number of degrees of freedom: ", fes.ndof)
   u,v = fes.TnT()
   a = BilinearForm(grad(u)*grad(v)*dx)
   f = LinearForm(fes)
   f += 32 * (y*(1-y)+x*(1-x)) * v * dx
   a.Assemble()
   f.Assemble()

We now construct an NGSolve preconditioner wrapping a `PETSc PC` object.
In particular, we begin considering an algebraic multigrid preconditioner constructed using `HYPRE`.
We will use the preconditioner just discussed inside NGSolve's own linear algebra solvers. ::

   from ngsPETSc.pc import *
   from ngsolve.krylovspace import CG
   pre = Preconditioner(a, "PETScPC", pc_type="hypre")
   gfu = GridFunction(fes)
   print("-------------------|HYPRE p={}|-------------------".format(order))
   gfu.vec.data = CG(a.mat, rhs=f.vec, pre=pre.mat, printrates=True)
   Draw(gfu)

We see that the HYPRE preconditioner is quite effective for the Poisson problem discretised using linear elements, but it is not as effective for higher-order elements.

.. list-table:: Preconditioners performance HYPRE
   :widths: auto
   :header-rows: 1

   * - Preconditioner
     - p=1
     - p=2
     - p=3
     - p=4
   * - HYPRE
     - 10 (1.57e-12)
     - 70 (2.22e-13)
     - 42 (3.96e-13)
     - 70 (1.96e-13)

To overcome this issue we will use a two-level additive Schwarz preconditioner.
In this case, we will use as fine space correction the inverse of the stiffness matrices associated with the patch of a vertex. ::

   def VertexPatchBlocks(mesh, fes):
    blocks = []
    freedofs = fes.FreeDofs()
    for v in mesh.vertices:
        vdofs = set()
        for el in mesh[v].elements:
            vdofs |= set(d for d in fes.GetDofNrs(el)
                         if freedofs[d])
        blocks.append(vdofs)
    return blocks

   blocks = VertexPatchBlocks(mesh, fes)
   dofs = BitArray(fes.ndof); dofs[:] = True
   blockjac = ASMPreconditioner(a.mat, dofs, blocks=blocks,
                                  solverParameters={"pc_type": "asm",
                                                    "sub_ksp_type": "preonly",
                                                    "sub_pc_type": "lu"})  

We now isolate the degrees of freedom associated with the vertices and construct a two-level additive Schwarz preconditioner, where the coarse space correction is the inverse of the stiffness matrices associated with the vertices. ::

   def VertexDofs(mesh, fes):
      vertexdofs = BitArray(fes.ndof)
      vertexdofs[:] = False
      for v in mesh.vertices:
         for d in fes.GetDofNrs(v):
            vertexdofs[d] = True
      vertexdofs &= fes.FreeDofs()
      return vertexdofs

   vertexdofs = VertexDofs(mesh, fes)
   preCoarse = PETScPreconditioner(a.mat, vertexdofs, solverParameters={"pc_type": "hypre"})
   pretwo = preCoarse + blockjac
   print("-------------------|Additive Schwarz p={}|-------------------".format(order))
   gfu.vec.data = CG(a.mat, rhs=f.vec, pre=pretwo, printrates=True)

We can see that the two-level additive Schwarz preconditioner where the coarse space correction is performed using HYPRE is more effective than using just HYPRE for higher-order elements.

.. list-table:: Preconditioners performance HYPRE and Two Level Additive Schwarz
   :widths: auto
   :header-rows: 1

   * - Preconditioner
     - p=1
     - p=2
     - p=3
     - p=4
   * - HYPRE
     - 10 (1.57e-12)
     - 70 (2.22e-13)
     - 42 (3.96e-13)
     - 70 (1.96e-13)
   * - Additive Schwarz
     - 44 (1.96e-12)
     - 45 (1.28e-12)
     - 45 (1.29e-12)
     - 45 (1.45e-12)
       
