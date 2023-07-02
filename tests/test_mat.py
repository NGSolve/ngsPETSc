'''
This module test that matrix class
'''
from ngsolve import Mesh, H1, BilinearForm, grad, dx
from netgen.geom2d import unit_square
import netgen.meshing as ngm

from mpi4py.MPI import COMM_WORLD

from ngsPETSc import Mat

def test_poisson_mat():
    '''
    Testing that the matrix for the Poisson problem is correctly exported.
    '''
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    fes = H1(mesh, order=1, dirichlet="left|right|top|bottom")
    u,v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    Mat(a.mat, fes.FreeDofs())
