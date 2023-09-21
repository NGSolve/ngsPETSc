'''
This module test the snes class
'''
from ngsolve import Mesh, H1, BilinearForm, GridFunction
from ngsolve import grad, dx, x,y
from netgen.geom2d import unit_square
import netgen.meshing as ngm

from mpi4py.MPI import COMM_WORLD

from ngsPETSc import NonLinearSolver

def test_snes_newtonls():
    '''
    Testing the mapping from NGSolve Vec to PETSc Vec
    '''
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    fes = H1(mesh, order=3, dirichlet="left|right|top|bottom")
    u,v = fes.TnT()
    a = BilinearForm(fes)
    a += (grad(u) * grad(v) + 1/3*u**3*v- 10 * v)*dx
    
    solver = NonLinearSolver(fes, a=a, solverParameters={"snes_type": "newtonls", "snes_monitor": ""})
    gfu0 = GridFunction(fes)
    gfu0.Set((x*(1-x))**4*(y*(1-y))**4) # initial guess
    solver.solve(gfu0)

def test_snes_lbfgs():
    '''
    Testing the mapping from NGSolve Vec to PETSc Vec
    '''
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    fes = H1(mesh, order=3, dirichlet="left|right|top|bottom")
    u,v = fes.TnT()
    a = BilinearForm(fes)
    a += (grad(u) * grad(v) + 1/3*u**3*v- 10 * v)*dx
    
    solver = NonLinearSolver(fes, a=a, solverParameters={"snes_type": "qn", "snes_monitor": ""})
    gfu0 = GridFunction(fes)
    gfu0.Set((x*(1-x))**4*(y*(1-y))**4) # initial guess
    solver.solve(gfu0)
if __name__ == '__main__':
    test_snes_lbfgs()
    test_snes_newtonls()
