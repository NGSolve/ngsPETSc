'''
This module test the eps class
'''
from math import sqrt, pi
import pytest

from ngsolve import Mesh, H1, BilinearForm, grad, Integrate
from ngsolve import x, y, dx, sin
from netgen.geom2d import SplineGeometry
import netgen.meshing as ngm

from mpi4py.MPI import COMM_WORLD

from ngsPETSc import Eigensolver

def test_eps_ghepi_eigvals():
    '''
    Testing the mapping PETSc KSP using MUMPS
    '''
    exact = [2,5,5,8]
    if COMM_WORLD.rank == 0:
        geo = SplineGeometry()
        geo.AddRectangle((0,0),(pi,pi),bc="bnd")
        mesh = Mesh(geo.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    fes = H1(mesh, order=3, dirichlet="bnd")
    u,v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx, symmetric=True).Assemble()
    m = BilinearForm(-1*u*v*dx, symmetric=True).Assemble()
    solver = Eigensolver((m, a), fes, 4, solverParameters={"eps_type":"arnoldi",
                                          "eps_smallest_magnitude":None,
                                          "eps_tol": 1e-6,
                                          "eps_target": 2,
                                          "st_type": "sinvert",
                                          "st_pc_type": "lu"})
    solver.solve()
    assert solver.nconv >= len(exact)
    for i in range(solver.nconv):
        assert abs(solver.eigenValue(i)-exact[i]) < 1e-4

@pytest.mark.mpi_skip()
def test_eps_ghep_eigfuncs():
    '''
    Testing the mapping PETSc KSP using MUMPS
    '''
    if COMM_WORLD.rank == 0:
        geo = SplineGeometry()
        geo.AddRectangle((0,0),(pi,pi),bc="bnd")
        mesh = Mesh(geo.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    fes = H1(mesh, order=3, dirichlet="bnd")
    u,v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx, symmetric=True).Assemble()
    m = BilinearForm(-1*u*v*dx, symmetric=True).Assemble()
    solver = Eigensolver((m,a), fes, 4, solverParameters={"eps_type":"arnoldi",
                                          "eps_smallest_magnitude":None,
                                          "eps_tol": 1e-6,
                                          "eps_target": 2,
                                          "st_type": "sinvert",
                                          "st_pc_type": "lu"})
    solver.solve()
    exactEigenMode = sin(x)*sin(y)
    eigenMode, _ = solver.eigenFunction(0)
    point = mesh(pi/2, pi/2)
    eigenMode = (1/eigenMode(point))*eigenMode
    assert sqrt(Integrate((eigenMode-exactEigenMode)**2, mesh))<1e-4
