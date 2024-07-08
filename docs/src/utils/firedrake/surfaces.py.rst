Surface meshes 
==================
In the present section we will show how to generate a surface mesh using Open Cascade and how to solve the heat equation on it.
We will start by generating a surface mesh for a torus. ::

    from netgen.occ import *
    from netgen.meshing import MeshingStep
    from math import sin, cos, pi
    import netgen.gui

    def Curve(t): return Pnt(0, 3+1.5*cos(t), sin(t))
    n = 100
    pnts = [Curve(2*pi*t/n) for t in range(n+1)]

    spline = SplineApproximation(pnts)
    f = Face(Wire(spline))

    torus = f.Revolve(Axis((0,0,0), Z), 360)

    ngmesh = OCCGeometry(torus).GenerateMesh(maxh=0.3, perfstepsend=MeshingStep.MESHSURFACE)

Once we have generated a surface mesh using Netgen, we can simply pass it to Firedrake and using the :code:`curve_field` method to generate a curved surface mesh. ::

    from firedrake import *
    mesh = Mesh(Mesh(ngmesh).curve_field(3))

We begin defining the initial condition for the problem, in particular, we will consider a Gaussian bump centered at a point on the torus. ::

    x, y, z = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 3)
    u0 = assemble(interpolate(exp(-10*((x-0.400126)**2 + (y+4.22785)**2 + (z-0.550597)**2)),V))
    u = Function(V)
    u.interpolate(u0)
    out = VTKFile("output/heat.pvd")
    out.write(u, time=0.0)

We now use `Irksome<https://www.firedrakeproject.org/Irksome>_`, to describe the time-dependent problem we aim to solve, i.e. 

.. math::

    \begin{align*}
    \frac{\partial u}{\partial t} - \nabla \cdot \nabla u &= 0 \quad \text{in} \quad \Omega \times (0, T], \\
    u &= u_0 \quad \text{at} \quad t = 0.
    \end{align*}

::

    from irksome import Dt
    v = TestFunction(V)
    F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx

We now need to define the final time we want to reach, and the number of time steps we want to take and the solver parameters.
We will then construct a :code:`TimeStepper` that will take care of time-stepping the PDE. ::

    from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper
    MC = MeshConstant(mesh)
    T = 1.0
    N = 100
    dt = MC.Constant(T / N)
    t = MC.Constant(0.0)
    luparams = {"mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu"}
    butcher_tableau = GaussLegendre(1)
    ns = butcher_tableau.num_stages
    stepper = TimeStepper(F, butcher_tableau, t, dt, u,
                      solver_parameters=luparams)
    while (float(t) < T):
        if (float(t) + float(dt) > T):
            dt.assign(T - float(t))
        stepper.advance()
        print(float(t))
        t.assign(float(t) + float(dt))

        out.write(u, time=float(t))
    