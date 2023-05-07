'''
This module test that the environment is correctly been setup.
In particular it will test for: petsc4py, PETSc, NGSolve, Netgen and ngsPETSc
'''

from ngsolve import Mesh, H1, BilinearForm, grad, dx
from netgen.geom2d import unit_square

from ngsPETSc import Mat

def test_poisson_mat():
    '''
    Testing that the matrix for the Poisson problem is correctly exported.
    '''
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
    fes = H1(mesh, order=1, dirichlet="left|right|top|bottom")
    u,v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    Mat(a.mat, fes.FreeDofs())
    
