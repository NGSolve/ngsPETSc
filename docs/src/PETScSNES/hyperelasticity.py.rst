Non-linear simulation of a Hyperelastic beam
=============================================

We here consider a simple beam with a hyperelastic material model. In particular, we will assume the beam has energy:

.. math::

    E(u) := \int_{\Omega} \frac{\mu}{2} tr(\mathbb{C}-I)+ \frac{2\mu}{\lambda} \det(\mathbb{C})^{-\frac{\lambda}{2\mu}-1}\, dx + \int_{\partial \Omega} \vec{f} \cdot \vec{u} \, ds

where :math:`\mathbb{C} = F^T F` is the right Cauchy-Green tensor, :math:`F` is the deformation gradient, :math:`\mu` and :math:`\lambda` are the Lamé parameters, and :math:`\vec{f}` is the force applied to the top face of the beam.
A discretization of this energy leads to a non-linear problem that we solve using `PETSc SNES`. ::

    from ngsolve import *
    import netgen.gui
    from netgen.occ import *
    import netgen.meshing as ngm
    from mpi4py.MPI import COMM_WORLD

    if COMM_WORLD.rank == 0:
        d = 0.01
        box = Box ( (-d/2,-d/2,0), (d/2,d/2,0.1) ) + Box( (-d/2, -3*d/2,0.1), (d/2, 3*d/2, 0.1+d) )
        box.faces.Min(Z).name = "bottom"
        box.faces.Max(Z).name = "top"
        mesh = Mesh(OCCGeometry(box).GenerateMesh(maxh=0.05).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    
    E, nu = 210, 0.2
    mu  = E / 2 / (1+nu)
    lam = E * nu / ((1+nu)*(1-2*nu))

    def C(u):
        F = Id(u.dim) + Grad(u)
        return F.trans * F

    def NeoHooke (C):
        # return 0.5*mu*InnerProduct(C-Id(3), C-Id(3))
        return 0.5*mu*(Trace(C-Id(3)) + 2*mu/lam*Det(C)**(-lam/2/mu)-1)
    
    loadfactor = Parameter(1)
    force = loadfactor * CF ( (-y, x, 0) )

    fes = H1(mesh, order=3, dirichlet="bottom", dim=mesh.dim)
    u,v = fes.TnT()

    a = BilinearForm(fes, symmetric=True)
    a += Variation(NeoHooke(C(u)).Compile()*dx)
    a += ((Id(3)+Grad(u.Trace()))*force)*v*ds("top")

Once we have defined the energy and the weak form, we can solve the non-linear problem using `PETSc SNES`.
In particular, we will use a Newton method with line search, and precondition the linear solves with a direct solver. ::

    from ngsPETSc import NonLinearSolver
    gfu_petsc = GridFunction(fes)
    gfu_ngs = GridFunction(fes)
    gfu_ngs.vec[:] = 0; gfu_petsc.vec[:] = 0
    gfu_history_ngs = GridFunction(fes, multidim=0)
    gfu_history_petsc = GridFunction(fes, multidim=0)
    numsteps = 30
    for step in range(numsteps):
        print("step", step)
        loadfactor.Set(720*step/numsteps)
        solver = NonLinearSolver(fes, a=a, objective=False,
                                 solverParameters={"snes_type": "newtonls",
                                                   "snes_max_it": 200,
                                                   "snes_monitor": "",
                                                   "ksp_type": "preonly",
                                                   "pc_type": "lu",
                                                   "snes_linesearch_type": "basic",
                                                   "snes_linesearch_damping": 1.0,
                                                   "snes_linesearch_max_it": 100})
        gfu_petsc = solver.solve(gfu_petsc)
        solvers.Newton (a, gfu_ngs, printing=True, dampfactor=0.5)
        gfu_history_ngs.AddMultiDimComponent(gfu_ngs.vec)
        gfu_history_petsc.AddMultiDimComponent(gfu_petsc.vec)

We compare the performance of the `PETSc SNES` solvers and the one of `NGSolve` own implementation of Newton's method:

.. list-table:: Performance of different non-linear solvers
   :widths: auto
   :header-rows: 1

   * - Solver
     - non-linear iteration
   * - NGS Newton (damping=1.0)
     - Diverged
   * - NGS Newton (damping=0.5)
     - Diverged
   * - NGS Newton (damping=0.3)
     - 12
   * - PETSc SNES
     - 10

This suggests that while NGS non-linear solver when finely tuned performs as well as PETSc SNES, it is more sensitive to the choice of the damping factor. In this case, a damping factor of 0.3 was found to be the best choice.
