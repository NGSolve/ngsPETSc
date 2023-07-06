'''
This module test the vec class
'''
from ngsolve import Mesh, H1, BilinearForm, dx
from netgen.geom2d import unit_square
import netgen.meshing as ngm

from mpi4py.MPI import COMM_WORLD

from ngsPETSc import Matrix, VectorMapping

def test_vec_map_ngs_petsc():
    '''
    Testing the mapping from NGSolve Vec to PETSc Vec
    '''
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    fes = H1(mesh, order=1, dirichlet="left|right|top|bottom")
    u,v = fes.TnT()
    M = BilinearForm(u*v*dx).Assemble().mat
    ngsVec = M.CreateColVector()
    Map = VectorMapping(fes)
    Map.petscVec(ngsVec)

def test_vec_map_petsc_ngs():
    '''
    Testing the mapping from PETSc Vec to NGSolve Vec
    '''
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    fes = H1(mesh, order=1, dirichlet="left|right|top|bottom")
    u,v = fes.TnT()
    m = BilinearForm(u*v*dx).Assemble()
    M = Matrix(m.mat, fes.FreeDofs())
    petscVec = M.mat.createVecLeft()
    Map = VectorMapping(fes)
    Map.ngsVec(petscVec)
