'''
This module test the snes class
'''
from ngsolve import Mesh, VectorH1, BilinearForm, Variation, H1
from ngsolve import GridFunction, CoefficientFunction, Parameter
from ngsolve import InnerProduct, Grad, grad, dx, x,y, Id, Trace, Det, exp, log
from netgen.geom2d import unit_square
import netgen.meshing as ngm

from mpi4py.MPI import COMM_WORLD

import pytest

from ngsPETSc import NonLinearSolver

def test_snes_toy_newtonls():
    '''
    Testing ngsPETSc SNES wrap for a toy problem, using newtonls
    '''
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    fes = H1(mesh, order=1, dirichlet="left|right|top|bottom")
    u,v = fes.TnT()
    a = BilinearForm(fes)
    a += (grad(u) * grad(v) + 1/3*u**3*v- 10 * v)*dx
    solver = NonLinearSolver(fes, a=a, objective=False,
                             solverParameters={"snes_type": "newtonls", "snes_monitor": "",
                                               "pc_type": "none"})
    gfu0 = GridFunction(fes)
    gfu0.Set((x*(1-x))**4*(y*(1-y))**4) # initial guess
    solver.solve(gfu0)
    assert solver.snes.getConvergedReason() in [4,3,2]

def test_snes_toy_lbfgs():
    '''
    Testing ngsPETSc SNES wrap for a toy problem, using lbfgs
    '''
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    fes = H1(mesh, order=3, dirichlet="left|right|top|bottom")
    u,v = fes.TnT()
    a = BilinearForm(fes)
    a += (grad(u) * grad(v) + 1/3*u**3*v- 10 * v)*dx
    solver = NonLinearSolver(fes, a=a, objective=False,
                             solverParameters={"snes_type": "qn", "snes_monitor": "",
                                               "pc_type": "lu"})
    gfu0 = GridFunction(fes)
    gfu0.Set((x*(1-x))**4*(y*(1-y))**4) # initial guess
    solver.solve(gfu0)
    assert solver.snes.getConvergedReason() in [4,3,2]

@pytest.mark.mpi_skip()
def test_snes_elastic_beam_newtonls():
    '''
    Testing ngsPETSc SNES wrap for NeoHook energy minimisation, using newtonls
    This test run only in serial becasue variational energy as objective only works
    in serial, please use objective=False in parallel. 
    '''
    from netgen.geom2d import SplineGeometry
    if COMM_WORLD.rank == 0:
        geo = SplineGeometry()
        pnums = [ geo.AddPoint (x,y,maxh=0.01) for x,y in [(0,0), (1,0), (1,0.1), (0,0.1)] ]
        for p1,p2,bc in [(0,1,"bot"), (1,2,"right"), (2,3,"top"), (3,0,"left")]:
            geo.Append(["line", pnums[p1], pnums[p2]], bc=bc)
        mesh = Mesh(geo.GenerateMesh(maxh=0.05).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    # E module and poisson number:
    E, nu = 210, 0.2
    # Lamé constants:
    mu  = E / 2 / (1+nu)
    lam = E * nu / ((1+nu)*(1-2*nu))
    fes = VectorH1(mesh, order=2, dirichlet="left")
    #gravity:
    force = CoefficientFunction( (0,-1) )
    u,_ = fes.TnT()
    def Pow(a, b):
        return exp (log(a)*b)

    def NeoHook (C):
        return 0.5 * mu * (Trace(C-I) + 2*mu/lam * Pow(Det(C), -lam/2/mu) - 1)

    I = Id(mesh.dim)
    F = I + Grad(u)
    C = F.trans * F
    factor = Parameter(0.1)
    a = BilinearForm(fes, symmetric=True)
    a += Variation(NeoHook (C).Compile() * dx
                    -factor * (InnerProduct(force,u) ).Compile() * dx)
    solver = NonLinearSolver(fes, a=a,
                             solverParameters={"snes_type": "newtonls",
                                               "snes_max_it": 10,
                                               "snes_monitor": "",
                                               "pc_type": "lu"})
    gfu0 = GridFunction(fes)
    gfu0.Set((0,0)) # initial guess
    solver.solve(gfu0)
    assert solver.snes.getConvergedReason() in [4,3,2]


if __name__ == '__main__':
    test_snes_toy_lbfgs()
    test_snes_toy_newtonls()
    if COMM_WORLD.size == 1:
        test_snes_elastic_beam_newtonls()
