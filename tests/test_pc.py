'''
This module test the pc class
'''
from math import sqrt

from ngsolve import Mesh, H1, BilinearForm, LinearForm, grad, Integrate
from ngsolve import x, y, dx, Preconditioner, GridFunction
from ngsolve.solvers import CG
from netgen.geom2d import unit_square
import netgen.meshing as ngm

from mpi4py.MPI import COMM_WORLD

from ngsPETSc import pc #, PETScPreconditioner

def test_pc():
    '''
    Testing the pc has registered function to register preconditioners
    '''
    assert hasattr(pc,"createPETScPreconditioner")

def test_pc_gamg():
    '''
    Testing the PETSc GAMG solver
    '''
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    fes = H1(mesh, order=4, dirichlet="left|right|top|bottom")
    u,v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx)
    a.Assemble()
    pre = Preconditioner(a, "PETScPC", pc_type="gamg")
    # pre = PETScPreconditioner(a.mat, fes.FreeDofs(), solverParameters={'pc_type': 'gamg'})
    f = LinearForm(fes)
    f += 32 * (y*(1-y)+x*(1-x)) * v * dx
    f.Assemble()
    gfu = GridFunction(fes)
    gfu.vec.data = CG(a.mat, rhs=f.vec, pre=pre, printrates=mesh.comm.rank==0)
    exact = 16*x*(1-x)*y*(1-y)
    assert sqrt(Integrate((gfu-exact)**2, mesh))<1e-4

if __name__ == '__main__':
    test_pc()
    test_pc_gamg()
