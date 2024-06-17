'''
This module test the ksp class
'''
from math import sqrt

from ngsolve import Mesh, H1, BilinearForm, LinearForm, grad, Integrate
from ngsolve import x, y, dx, GridFunction
from netgen.geom2d import unit_square
import netgen.meshing as ngm

from mpi4py.MPI import COMM_WORLD

from ngsPETSc import KrylovSolver

def test_ksp_preonly_lu():
    '''
    Testing the mapping PETSc KSP using a direct solver
    '''
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    fes = H1(mesh, order=3, dirichlet="left|right|top|bottom")
    u,v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx)
    solver = KrylovSolver(a, fes.FreeDofs(),
                          solverParameters={'ksp_type': 'preonly',
                                            'ksp_monitor': '',
                                            'pc_type': 'lu',})
    f = LinearForm(fes)
    f += 32 * (y*(1-y)+x*(1-x)) * v * dx
    f.Assemble()
    gfu = GridFunction(fes)
    solver.solve(f.vec, gfu.vec)
    exact = 16*x*(1-x)*y*(1-y)
    assert sqrt(Integrate((gfu-exact)**2, mesh))<1e-4

def test_ksp_cg_gamg():
    '''
    Testing the mapping PETSc KSP using GAMG
    '''
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    fes = H1(mesh, order=3, dirichlet="left|right|top|bottom")
    u,v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    solver = KrylovSolver(a,fes.FreeDofs(),
                          solverParameters={'ksp_type': 'cg',
                                            'ksp_monitor': '',
                                            'pc_type': 'gamg'})
    f = LinearForm(fes)
    f += 32 * (y*(1-y)+x*(1-x)) * v * dx
    f.Assemble()
    gfu = GridFunction(fes)
    solver.solve(f.vec, gfu.vec)
    exact = 16*x*(1-x)*y*(1-y)
    assert sqrt(Integrate((gfu-exact)**2, mesh))<1e-4

if __name__ == '__main__':
    test_ksp_cg_gamg()
    test_ksp_preonly_lu()
