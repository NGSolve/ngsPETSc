"""
This module test the utils.fenicsx class
"""

import pytest
from packaging.version import Version


def test_square_netgen():
    """
    Testing FEniCSx interface with Netgen generating a square mesh
    """
    try:
        from mpi4py import MPI
        import ngsPETSc.utils.fenicsx as ngfx
        from dolfinx.io import XDMFFile
    except ImportError:
        pytest.skip("DOLFINx unavailable, skipping FENICSx test")

    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1))
    geoModel = ngfx.GeometricModel(geo, MPI.COMM_WORLD)
    domain, _, _ = geoModel.model_to_mesh(hmax=0.1)
    with XDMFFile(domain.comm, "XDMF/mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)


def test_poisson_netgen():
    """
    Testing FEniCSx interface with Netgen generating a square mesh
    """
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
    geo.AddRectangle((0, 0), (np.pi, np.pi))
    geoModel = ngfx.GeometricModel(geo, MPI.COMM_WORLD)
    msh, _, _ = geoModel.model_to_mesh(hmax=0.1)
    V = fem.functionspace(msh, ("Lagrange", 2))  # pylint: disable=E1120
    facetsLR = mesh.locate_entities_boundary(
        msh,
        dim=(msh.topology.dim - 1),
        marker=lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], np.pi)),
    )
    facetsTB = mesh.locate_entities_boundary(
        msh,
        dim=(msh.topology.dim - 1),
        marker=lambda x: np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], np.pi)),
    )
    facets = np.append(facetsLR, facetsTB)
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = ufl.exp(ufl.sin(x[0]) * ufl.sin(x[1]))
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx
    # Backward compatibility
    if Version(dfx_version) < Version("0.10.0"):
        options = {}
    else:
        options = {"petsc_options_prefix": "test_solver"}
    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        **options,
    )

    problem.solve()


@pytest.mark.parametrize("order", [1, 2, 3])
def test_markers(order):
    """Test cell and facet markers."""
    try:
        from mpi4py import MPI
        import ngsPETSc.utils.fenicsx as ngfx
        import dolfinx
        import ufl
        from netgen.occ import OCCGeometry, WorkPlane, Glue
        import numpy as np
    except ImportError:
        pytest.skip("DOLFINx unavailable, skipping FENICSx test.")

    wp = WorkPlane()
    square = wp.Rectangle(1, 1).Face()
    disk = wp.Circle(0.2, 0.2, 0.1).Face()
    disk.edges.name = "circle"
    disk.faces.name = "circle"
    shape = Glue([square, disk])
    geo = OCCGeometry(shape, dim=2)
    geoModel = ngfx.GeometricModel(geo, MPI.COMM_WORLD)
    gm = dolfinx.mesh.GhostMode.shared_facet
    partitioner = dolfinx.mesh.create_cell_partitioner(gm)
    _, (ct, _), region_map = geoModel.model_to_mesh(
        hmax=0.02, partitioner=partitioner
    )
    curved_domain = geoModel.curveField(order)

    steel_circle = region_map[(2, "circle")]

    _, (ct_refined, ft_refined) = geoModel.refineMarkedElements(
        ct.dim, ct.indices[np.isin(ct.values, steel_circle)]
    )
    curved_domain = geoModel.curveField(order)
    # Integrate over interior marked interface
    dS = ufl.Measure("dS", domain=curved_domain, subdomain_data=ft_refined)
    crack_integer_marker = region_map[(1, "circle")]
    local_interface = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(1 * dS(crack_integer_marker))
    )
    interface = curved_domain.comm.allreduce(local_interface, op=MPI.SUM)

    # Integrate over marked subdomain
    dx = ufl.Measure("dx", domain=curved_domain, subdomain_data=ct_refined)
    steel_circle = region_map[(2, "circle")]
    local_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * dx(steel_circle)))
    area = curved_domain.comm.allreduce(local_area, op=MPI.SUM)

    if order == 1:
        tol = 5e-4
    else:
        tol = 1e-6
    assert np.isclose(interface, 2 * np.pi * 0.1, atol=tol, rtol=tol)
    assert np.isclose(area, np.pi * 0.1**2, atol=tol, rtol=tol)

def test_refine():
    """Test cell and facet markers."""
    try:
        from mpi4py import MPI
        import ngsPETSc.utils.fenicsx as ngfx
        import dolfinx
        import ufl
        from netgen.csg import Sphere, Pnt, CSGeometry
        import numpy as np
    except ImportError:
        pytest.skip("DOLFINx unavailable, skipping FENICSx test.")

    center0 = Pnt(0,0,0)
    center1 = Pnt(0.2,0.2,0)
    radius0 = 1
    radius1 = 0.3
    sphere0 = Sphere(center0, radius0)
    sphere0.bc("Outer")
    sphere0.mat("Material")
    sphere1 = Sphere(center1, radius1)
    sphere1.bc("Inner")
    shape = sphere0 - sphere1
    geo = CSGeometry()
    geo.Add(shape)

    geoModel = ngfx.GeometricModel(geo, MPI.COMM_WORLD)

    gm = dolfinx.mesh.GhostMode.shared_facet
    partitioner = dolfinx.mesh.create_cell_partitioner(gm)
    mesh, (_, _), region_map = geoModel.model_to_mesh(
        hmax=0.1, partitioner=partitioner, gdim=3
    )
    # NOTE: IF the following is called, netgen segfaults at curve after refine
    #mesh = geoModel.curveField(2)
    def locate_facets(x):
        return np.isclose(x[0]**2 + x[1]**2 + x[2]**2, radius0)

    facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim-1, locate_facets)

    refined_mesh, (_, ft_refined) = geoModel.refineMarkedElements(
        mesh.topology.dim-1, facets)
    refined_mesh = geoModel.curveField(2)

    local_vol = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1*ufl.dx(domain=refined_mesh)))
    vol = refined_mesh.comm.allreduce(local_vol, op=MPI.SUM)

    ds = ufl.ds(domain=refined_mesh, subdomain_data=ft_refined)
    inner_area = dolfinx.fem.form(1*ds(region_map[(2, "Inner")]))
    outer_area = dolfinx.fem.form(1*ds(region_map[(2, "Outer")]))
    local_inner = dolfinx.fem.assemble_scalar(inner_area)
    local_outer = dolfinx.fem.assemble_scalar(outer_area)
    inner = refined_mesh.comm.allreduce(local_inner, op=MPI.SUM)
    outer = refined_mesh.comm.allreduce(local_outer, op=MPI.SUM)
    tol = 5e-5
    assert np.isclose(vol, 4/3*np.pi*radius0**3 - 4/3*np.pi*radius1**3)
    assert np.isclose(inner, 4*np.pi*radius1**2, rtol=tol)
    assert np.isclose(outer, 4*np.pi*radius0**2, rtol=tol)


if __name__ == "__main__":
    test_square_netgen()
    test_poisson_netgen()
    test_markers(2)
    test_refine()
