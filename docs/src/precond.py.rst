Using PETSc PC inside of NGSolve
=================================

In this tutorial we explore using `PETSc PC` as a preconditioner inside NGSolve preconditioning infrastructure.
Once again, we begin by creating a discretisation of the Poisson problem using H1 elements, in particular we consider the usual variational formulation
.. math::

   \text{find } u\in H^1_0(\Omega) \text{ s.t. } a(u,v) := \int_{\Omega} \nabla u\cdot \nabla v \; d\vec{x} = L(v) := \int_{\Omega} fv\; d\vec{x}\qquad v\in H^1_0(\Omega).

Such a discretisation can easily be constructed using NGSolve as follows: ::

   from ngsolve import *
   import netgen.gui
   import netgen.meshing as ngm
   from mpi4py.MPI import COMM_WORLD

   if COMM_WORLD.rank == 0:
      mesh = Mesh(unit_square.GenerateMesh(maxh=0.2).Distribute(COMM_WORLD))
   else:
      mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
   for _ in range(2):
      mesh.Refine()
   fes = H1(mesh, order=3, dirichlet="left|right|top|bottom")
   u,v = fes.TnT()
   a = BilinearForm(grad(u)*grad(v)*dx)
   f = LinearForm(fes)
   f += 32 * (y*(1-y)+x*(1-x)) * v * dx
   a.Assemble()
   f.Assemble()

We now consturct an NGSolve preconditioner wrapping a `PETSc PC`, in particular we will construct an Algebraic MultiGrid preconditioner using `HYPRE` and use the Krylov solver implemented inside NGSolve to solve the linear system. ::

   from ngsPETSc import pc
   from ngsolve.krylovspace import CG
   pre = Preconditioner(a, "PETScPC", pc_type="hypre")
   gfu = GridFunction(fes)
   gfu.vec.data = CG(a.mat, rhs=f.vec, pre=pre.mat, printrates=True)
   Draw(gfu)

We can use PETSc preconditioner as one of the building blocks of a more complex preconditioner. For example, we can construct an additive Schwarz preconditioner.
In this case, we will use as fine space correction, the inverse of the local matrices associated with the patch of a vertex. ::

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
   blockjac = a.mat.CreateBlockSmoother(blocks)
   gfu.vec.data = CG(a.mat, rhs=f.vec, pre=blockjac, printrates=True)
   Draw(gfu)

We now isolate the degrees of freedom associated with the vertices and construct a two-level additive Schwarz preconditioner, where the coarse space correction is the inverse of the local matrices associated with the vertices. ::

   def VertexDofs(mesh, fes):
      vertexdofs = BitArray(fes.ndof)
      vertexdofs[:] = False
      for v in mesh.vertices:
         for d in fes.GetDofNrs(v):
            vertexdofs[d] = True
      vertexdofs &= fes.FreeDofs()
      return vertexdofs

   vertexdofs = VertexDofs(mesh, fes)
   preCoarse = Preconditioner(a, "PETScPC", pc_type="hypre", restrictedTo=vertexdofs)
   pretwo = preCoarse.mat + blockjac
   gfu.vec.data = CG(a.mat, rhs=f.vec, pre=pretwo, printrates=True)


We can also use the PETSc preconditioner as an auxiliary space preconditioner.
Let us consdier the disctinuous Galerkin discretisation of the Poisson problem. ::

   fesDG = L2(mesh, order=3, dgjumps=True)
   u,v = fesDG.TnT()
   aDG = BilinearForm(fesDG)
   jump_u = u-u.Other(); jump_v = v-v.Other()
   n = specialcf.normal(2)
   mean_dudn = 0.5*n * (grad(u)+grad(u.Other()))
   mean_dvdn = 0.5*n * (grad(v)+grad(v.Other()))
   alpha = 4
   h = specialcf.mesh_size
   aDG = BilinearForm(fesDG)
   aDG += grad(u)*grad(v) * dx
   aDG += alpha*3**2/h*jump_u*jump_v * dx(skeleton=True)
   aDG += alpha*3**2/h*u*v * ds(skeleton=True)
   aDG += (-mean_dudn*jump_v -mean_dvdn*jump_u)*dx(skeleton=True)
   aDG += (-n*grad(u)*v-n*grad(v)*u)*ds(skeleton=True)
   fDG = LinearForm(fesDG)
   fDG += 1*v * dx
   aDG.Assemble()
   fDG.Assemble()

We can now use the PETSc PC assembled for the confroming Poisson problem as an auxiliary space preconditioner for the DG discretisation. ::

   from ngsPETSc import pc
   smoother = Preconditioner(aDG, "PETScPC", pc_type="sor")
   transform = fes.ConvertL2Operator(fesDG)
   preDG = transform @ pre.mat @ transform.T + smoother.mat
   gfuDG = GridFunction(fesDG)
   gfuDG.vec.data = CG(aDG.mat, rhs=fDG.vec, pre=preDG, printrates=True)
   Draw(gfuDG)