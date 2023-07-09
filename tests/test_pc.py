'''
This module test the pc class
'''
from math import sqrt
import pytest

from ngsolve import Mesh, H1, BilinearForm, LinearForm, grad, Integrate
from ngsolve import x, y, dx, Preconditioner, GridFunction, CGSolver
from ngsolve.solvers import CG
from netgen.geom2d import unit_square
import netgen.meshing as ngm

from mpi4py.MPI import COMM_WORLD

from ngsPETSc import pc
#import ngsolve.ngs2petsc as n2p

def test_pc():
    assert hasattr(pc,"createPETScPreconditioner")

@pytest.mark.mpi
def test_pc_cg_lu():
    '''
    Testing the MUMPS PETSc PC
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
    f = LinearForm(fes)
    f += 32 * (y*(1-y)+x*(1-x)) * v * dx
    f.Assemble()
    gfu = GridFunction(fes)
    gfu.vec.data = CG(a.mat, rhs=f.vec, pre=pre, printrates=mesh.comm.rank==0)
    exact = 16*x*(1-x)*y*(1-y)
    assert sqrt(Integrate((gfu-exact)**2, mesh))<1e-4