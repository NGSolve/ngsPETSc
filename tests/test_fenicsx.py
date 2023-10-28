'''
This module test the utils.fenicsx class
'''
import pytest

def test_square_netgen():
    '''
    Testing FEniCSx interface with Netgen generating a square mesh
    '''
    try:
        from mpi4py import MPI
        import ngsPETSc.utils.fenicsx as ngfx
        from dolfinx.io import XDMFFile
    except ImportError:
        pytest.skip(msg="DOLFINx unavailable, skipping FENICSx test")

    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle((0,0),(1,1))
    geoModel = ngfx.GeometricModel(geo, MPI.COMM_WORLD)
    domain  =geoModel.model_to_mesh(hmax=0.1)
    with XDMFFile(domain.comm, "XDMF/mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)

def test_poisson_netgen():
    '''
    Testing FEniCSx interface with Netgen generating a square mesh
    '''
    try:
        import numpy as np
        import ufl
        from dolfinx import fem, mesh
        from dolfinx.fem.petsc import LinearProblem
        from ufl import dx, grad, inner
        from mpi4py import MPI
        from petsc4py.PETSc import ScalarType
        import ngsPETSc.utils.fenicsx as ngfx
    except ImportError:
        pytest.skip(msg="DOLFINx unavailable, skipping FENICSx test")

    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle((0,0),(np.pi,np.pi))
    geoModel = ngfx.GeometricModel(geo, MPI.COMM_WORLD)
    msh  =geoModel.model_to_mesh(hmax=0.1)
    V = fem.FunctionSpace(msh, ("Lagrange", 2))
    facetsLR = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1),
             marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
             np.isclose(x[0], np.pi)))
    facetsTB = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1),
             marker=lambda x: np.logical_or(np.isclose(x[1], 0.0),
             np.isclose(x[1], np.pi)))
    facets = np.append(facetsLR,facetsTB)
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = ufl.exp(ufl.sin(x[0])*ufl.sin(x[1]))
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx
    problem = LinearProblem(a, L, bcs=[bc],
              petsc_options={"ksp_type": "cg", "pc_type": "qr"})
    problem.solve()

if __name__ == "__main__":
    test_square_netgen()
    test_poisson_netgen()
