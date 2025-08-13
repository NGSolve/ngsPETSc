'''
This module test the utils.fenicsx class
'''
import pytest
from packaging.version import Version

def test_square_netgen():
    '''
    Testing FEniCSx interface with Netgen generating a square mesh
    '''
    try:
        from mpi4py import MPI
        import ngsPETSc.utils.fenicsx as ngfx
        from dolfinx.io import XDMFFile
    except ImportError:
        pytest.skip("DOLFINx unavailable, skipping FENICSx test")

    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle((0,0),(1,1))
    geoModel = ngfx.GeometricModel(geo, MPI.COMM_WORLD)
    domain, _, _  = geoModel.model_to_mesh(hmax=0.1)
    with XDMFFile(domain.comm, "XDMF/mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)

def test_poisson_netgen():
    '''
    Testing FEniCSx interface with Netgen generating a square mesh
    '''
    try:
        import numpy as np
        import ufl
        from dolfinx import __version__ as dfx_version
        from dolfinx import fem, mesh
        from dolfinx.fem.petsc import LinearProblem
        from ufl import dx, grad, inner
        from mpi4py import MPI
        from petsc4py.PETSc import ScalarType
        import ngsPETSc.utils.fenicsx as ngfx
    except ImportError:
        pytest.skip("DOLFINx unavailable, skipping FENICSx test")

    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle((0,0),(np.pi,np.pi))
    geoModel = ngfx.GeometricModel(geo, MPI.COMM_WORLD)
    msh, _, _  = geoModel.model_to_mesh(hmax=0.1)
    V = fem.functionspace(msh, ("Lagrange", 2)) #pylint: disable=E1120
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
    # Backward compatibility
    if Version(dfx_version) < Version("0.10.0"):
        options = {}
    else:
        options = {"petsc_options_prefix": "test_solver"}
    problem = LinearProblem(a, L, bcs=[bc],
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                            **options)

    problem.solve()


def test_markers():
    """Test cell and facet markers."""
    try:
        from mpi4py import MPI
        import ngsPETSc.utils.fenicsx as ngfx
        import dolfinx
        import ufl
        from netgen.occ import OCCGeometry, WorkPlane, Glue
        import numpy as np
    except ImportError:
        pytest.skip("DOLFINx unavailable, skipping FENICSx test")

    wp = WorkPlane()
    square = wp.Rectangle(1,1).Face()
    disk = wp.Circle(0.2, 0.2, 0.1).Face()
    disk.edges.name = "circle"
    disk.faces.name = "circle"
    shape = Glue([square, disk])
    geo = OCCGeometry(shape, dim=2)
    geoModel = ngfx.GeometricModel(geo, MPI.COMM_WORLD)
    domain, (ct, ft), region_map = geoModel.model_to_mesh(hmax=0.015)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=ft)

    crack_integer_marker = region_map[(1, "circle")]

    local_interface = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1*dS(crack_integer_marker)))
    interface = domain.comm.allreduce(local_interface, op=MPI.SUM)
    assert np.isclose(interface, 2*np.pi*0.1, atol=5e-4, rtol=5e-4)

    dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)
    steel_circle = region_map[(2, "circle")]
    local_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1*dx(steel_circle)))
    area = domain.comm.allreduce(local_area, op=MPI.SUM)
    assert np.isclose(area, np.pi*0.1**2, atol=5e-4, rtol=5e-4)

if __name__ == "__main__":
    test_square_netgen()
    test_poisson_netgen()
    test_markers()
