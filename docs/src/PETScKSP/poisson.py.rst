Solving the Poisson problem with different preconditioning strategies
=======================================================================

In this tutorial, we explore using `PETSc KSP` as an out-of-the-box solver inside NGSolve.
We will focus our attention on the Poisson problem, and we will consider different solvers and preconditioning strategies.
In particular, we will show how to use `MUMPS` as a direct solver, `ILU` as an incomplete LU factorisation, `AMG` as an Algebraic MultiGrid preconditioner, and `BDDC` as a Balancing Domain Decomposition preconditioner.

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

Now that we have a discretisation of the we use ngsPETSc `KrylovSolver` to solve the linear system.
`KrylovSolver` wraps the PETSc KSP object.
We begin showing how to use an LU factorisation as a direct solver, in particular, we use `MUMPS` to perform the factorisation in parallel.
Let us discuss the solver options, i.e. the flag :code:`ksp_type` set to :code:`preonly` enforces the use of a direct solver, :code:`pc_type` enforces that we use a direct LU factorisation, lastly :code:`pc_factor_mat_solver_type` enforces the use of `MUMPS`. ::

    from ngsPETSc import KrylovSolver
    solver = KrylovSolver(a, fes.FreeDofs(), 
                          solverParameters={"ksp_type": "preonly", 
                                            "pc_type": "lu",
                                            "pc_factor_mat_solver_type": "mumps"})
    gfu = GridFunction(fes)
    solver.solve(f.vec, gfu.vec)
    exact = 16*x*(1-x)*y*(1-y)
    print ("LU L2-error:", sqrt (Integrate ( (gfu-exact)*(gfu-exact), mesh)))
    Draw(gfu)

We can also use an interative solver with an incomplete LU factorisation as a preconditioner.
We have now switched to an interative solver setting the :code:`ksp_type` flag to :code:`cg`, while we enforce the use of an incomplete LU using once again the flag :code:`pc_type`.
We have also added the flag :code:`ksp_monitor` to view the residual at each linear iteration. ::

    if COMM_WORLD.Get_size() == 1:
      solver = KrylovSolver(a, fes.FreeDofs(), 
                            solverParameters={"ksp_type": "cg",
                                              "ksp_monitor": "",
                                              "pc_type": "ilu",
                                              "pc_factor_mat_solver_type": "petsc"})
      gfu = GridFunction(fes)
      solver.solve(f.vec, gfu.vec)
      print ("ILU L2-error:", sqrt (Integrate ( (gfu-exact)*(gfu-exact), mesh)))
      Draw(gfu)
    else:
      print("ILU preconditioner is not available in parallel")

.. list-table:: Preconditioners performance
   :widths: auto
   :header-rows: 1

   * - Preconditioner
     - Iterations
   * - PETSc ILU
     - 166

We can also use an Algebraic MultiGrid preconditioner, in particular we PETSc own implementation of AMG.
We do this changing the falg :code:`pc_type` to :code:`gamg` ::

    solver = KrylovSolver(a, fes.FreeDofs(), 
                          solverParameters={"ksp_type": "cg", 
                                            "ksp_monitor": "",
                                            "pc_type": "gamg"})
    gfu = GridFunction(fes)
    solver.solve(f.vec, gfu.vec)
    print ("GAMG L2-error:", sqrt (Integrate ( (gfu-exact)*(gfu-exact), mesh)))
    Draw(gfu)

.. list-table:: Preconditioners performance
   :widths: auto
   :header-rows: 1

   * - Preconditioner
     - Iterations
   * - PETSc ILU
     - 166
   * - PETSc GAMG
     - 35

We can also use PETSc `BDDC` preconditioner.
Once again we will enforce this option via the flag :code:`pc_type` flag.
We will also use the flag :code:`ksp_rtol` to obtain a more accurate solution of the linear system. ::

    solver = KrylovSolver(a, fes.FreeDofs(), 
                          solverParameters={"ksp_type": "cg", 
                                            "ksp_monitor": "",
                                            "pc_type": "bddc",
                                            "ngs_mat_type": "is",
                                            "ksp_rtol": 1e-10})
    gfu = GridFunction(fes)
    solver.solve(f.vec, gfu.vec)
    print ("BDDC L2-error:", sqrt (Integrate ( (gfu-exact)*(gfu-exact), mesh)))
    Draw(gfu)

.. list-table:: Preconditioners performance
   :widths: auto
   :header-rows: 1

   * - Preconditioner
     - Iterations
   * - PETSc ILU
     - 166
   * - PETSc GAMG
     - 35
   * - PETSc BDDC (N=2)
     - 10
   * - PETSc BDDC (N=4)
     - 12
   * - PETSc BDDC (N=6)
     - 14

We see that for an increasing number of subdomains :math:`N` the number of iterations increases.
Notice that in all the cases we have considered so far ngsPETSc :code:`KrylovSolver` had to create a PETSc matrix from the NGSolve matrix in order to assemble the required preconditioners.
If we have already some knowledge of the preconditioner we want to use, we can use :code:`KrylovSolver` in a matrix-free fashion.
This will result in a faster setup time and less memory usage. 
For example, we can use :code:`KrylovSolver`: in a matrix-free fashion with the element-wise BDDC preconditioner implemented in NGSolve.
Notice that because in NGSolve one needs to "register" the preconditioner before assembling the associated matrix we will have to redefine the :code:`BilinearForm` associated with the Poisson problem. :: 


    a = BilinearForm(grad(u)*grad(v)*dx)
    el_bddc = Preconditioner(a, "bddc")
    a.Assemble()
    solver = KrylovSolver(a.mat, fes.FreeDofs(), p=el_bddc.mat,
                          solverParameters={"ksp_type": "cg", 
                                            "ksp_monitor": "",
                                            "pc_type": "mat",
                                            "ngs_mat_type": "python",
                                            "ksp_rtol": 1e-10})
    gfu = GridFunction(fes)
    solver.solve(f.vec, gfu.vec)
    print ("Element-wise BDDC L2-error:", sqrt (Integrate ( (gfu-exact)*(gfu-exact), mesh)))
    Draw(gfu)