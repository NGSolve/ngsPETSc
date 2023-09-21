'''
This module test the matrix class
'''
from ngsolve import Mesh, H1, VectorH1, BilinearForm, grad, div, dx
from netgen.geom2d import unit_square
import netgen.meshing as ngm

from mpi4py.MPI import COMM_WORLD

from ngsPETSc import Matrix

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
    Matrix(a.mat, fes)

def test_poisson_mat_filling():
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
    M = Matrix(a.mat, fes)
    Matrix(a.mat, fes, petscMat=M.mat)

def test_nonsquare_mat():
    '''
    Testing that a nonsquare matrix is correctly exported.
    '''
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))

    V = VectorH1(mesh, order=2, dirichlet="left|right|top|bottom")
    Q = H1(mesh, order=1)

    u,_ = V.TnT()
    _,q = Q.TnT()

    b = BilinearForm(trialspace=V, testspace=Q)
    b += div(u)*q*dx
    b.Assemble()
    Matrix(b.mat, (Q, V))

if __name__ == '__main__':
    test_poisson_mat()
    test_poisson_mat_filling()
    test_nonsquare_mat()
