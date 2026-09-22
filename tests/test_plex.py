'''
This module tests the plex class
'''

import numpy as np
from mpi4py import MPI

try:
    from netgen.csg import unit_cube
    from netgen.geom2d import unit_square
    from ngsolve import VOL, Mesh
except ImportError:
    Mesh = None
    VOL = unit_square = unit_cube = None

import pytest
from petsc4py import PETSc

from ngsPETSc import MeshMapping


def _plex_number_of_points(plex, h=0, local=False):
    points = plex.getHeightStratum(h)
    npoints = points[1] - points[0]
    if not local:
        npoints = plex.getComm().tompi4py().allreduce(npoints)
    return npoints

@pytest.mark.mpi_skip
@pytest.mark.ngsolve_skip
def test_ngs_plex_2d():
    '''
    Testing the conversion from NGSolve mesh to PETSc DMPlex
    for a two dimensional simplex mesh
    '''
    mesh = Mesh(unit_square.GenerateMesh(maxh=1.))
    meshMap = MeshMapping(mesh)
    plex = meshMap.petscPlex
    assert _plex_number_of_points(plex) == 2

@pytest.mark.mpi_skip
@pytest.mark.ngsolve_skip
def test_plex_ngs_2d():
    '''
    Testing the conversion from PETSc DMPlex to NGSolve mesh
    for a two dimensional simplex mesh
    '''
    cells = [[0, 1, 3], [1, 3, 4], [1, 2, 4], [2, 4, 5],
             [3, 4, 6], [4, 6, 7], [4, 5, 7], [5, 7, 8]]
    cooridinates = [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
              [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
              [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]]
    plex = PETSc.DMPlex().createFromCellList(2, cells,
                                             cooridinates,
                                             comm=PETSc.COMM_WORLD)
    nc = _plex_number_of_points(plex, local=True)
    meshMap = MeshMapping(plex)
    assert Mesh (meshMap.ngMesh).GetNE(VOL) == nc


@pytest.mark.parallel([1, 2])
def test_plex_to_netgen_preserves_geometry_and_face_region_numbers():
    """Preserve geometry and face-region numbers after DMPlex redistribution."""
    comm = PETSc.COMM_WORLD
    plex = PETSc.DMPlex().createBoxMesh([2, 2], simplex=False, comm=comm)
    transform = PETSc.DMPlexTransform().create(comm=comm)
    transform.setType(PETSc.DMPlexTransformType.REFINETOSIMPLEX)
    transform.setDM(plex)
    transform.setUp()
    plex = transform.apply(plex)
    plex.distribute(overlap=0)

    ngmesh = MeshMapping(plex).ngMesh
    vStart, vEnd = plex.getDepthStratum(0)
    local_coordinates = plex.getCoordinatesLocal().getArray().reshape(-1, 2)
    assert local_coordinates.shape[0] == vEnd - vStart
    fStart, fEnd = plex.getHeightStratum(1)
    boundary_faces = [point for point in range(fStart, fEnd)
                      if plex.getLabelValue("Face Sets", point) >= 0]
    plex_boundary_coords = sorted(
        tuple(sorted(tuple(local_coordinates[v - vStart])
                     for v in plex.getCone(point)))
        for point in boundary_faces)
    local_ids = plex.getLabelIdIS("Face Sets").indices
    local_gap = len(local_ids) > 0 and not np.array_equal(
        local_ids, np.arange(1, local_ids[-1] + 1))
    has_gap = comm.tompi4py().allreduce(local_gap, op=MPI.LOR)
    if comm.getSize() > 1:
        assert has_gap
    expected = sorted(plex.getLabelValue("Face Sets", point)
                      for point in boundary_faces)
    elements = ngmesh.Elements1D().NumPy()
    actual = sorted(map(int, elements["index"]))
    ng_coordinates = np.array([ngmesh.Points()[point].p[:2]
                               for point in range(1, len(ngmesh.Points()) + 1)])
    netgen_boundary_coords = sorted(
        tuple(sorted(tuple(ng_coordinates[vertex - 1])
                     for vertex in nodes[:2]))
        for nodes in elements["nodes"])
    np.testing.assert_allclose(netgen_boundary_coords, plex_boundary_coords)
    assert actual == expected

@pytest.mark.mpi_skip
@pytest.mark.ngsolve_skip
def test_ngs_plex_3d():
    '''
    Testing the conversion from NGSolve mesh to PETSc DMPlex
    for a three dimensional simplex mesh
    '''
    mesh = Mesh(unit_cube.GenerateMesh(maxh=1.))
    meshMap = MeshMapping(mesh)
    plex = meshMap.petscPlex
    assert _plex_number_of_points(plex) == 12

@pytest.mark.mpi_skip
@pytest.mark.ngsolve_skip
def test_plex_ngs_3d():
    '''
    Testing the conversion from PETSc DMPlex to NGSolve mesh
    for a three dimensional simplex mesh
    '''
    cells = [[0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 6, 7],
             [0, 1, 3, 7], [0, 1, 5, 7], [0, 4, 5, 7]]
    cooridinates = [[0., 0., 0.], [1., 0., 0.],
                    [0., 1., 0.], [1., 1., 0.],
                    [0., 0., 1.], [1., 0., 1.],
                    [0., 1., 1.], [1., 1., 1.]]
    plex = PETSc.DMPlex().createFromCellList(3, cells,
                                             cooridinates,
                                             comm=PETSc.COMM_WORLD)
    nc = _plex_number_of_points(plex, local=True)
    meshMap = MeshMapping(plex)
    assert Mesh(meshMap.ngMesh).GetNE(VOL) == nc

@pytest.mark.mpi_skip
@pytest.mark.ngsolve_skip
def test_plex_transform_alfeld_2d():
    '''
    Testing the use of the PETSc Alfeld transform
    on a NGSolve mesh.
    '''
    mesh = Mesh(unit_square.GenerateMesh(maxh=1.))
    meshMap = MeshMapping(mesh)
    tr = PETSc.DMPlexTransform().create(comm=PETSc.COMM_WORLD)
    tr.setType(PETSc.DMPlexTransformType.REFINEALFELD)
    tr.setDM(meshMap.petscPlex)
    tr.setUp()
    newplex = tr.apply(meshMap.petscPlex)
    nc = _plex_number_of_points(newplex, local=True)
    meshMap = MeshMapping(newplex)
    assert Mesh(meshMap.ngMesh).GetNE(VOL) == nc

@pytest.mark.mpi_skip
@pytest.mark.ngsolve_skip
def test_plex_transform_alfeld_3d():
    '''
    Testing the use of the PETSc Alfeld transform
    on a NGSolve mesh.
    '''
    mesh = Mesh(unit_cube.GenerateMesh(maxh=1.))
    meshMap = MeshMapping(mesh)
    tr = PETSc.DMPlexTransform().create(comm=PETSc.COMM_WORLD)
    tr.setType(PETSc.DMPlexTransformType.REFINEALFELD)
    tr.setDM(meshMap.petscPlex)
    tr.setUp()
    newplex = tr.apply(meshMap.petscPlex)
    nc = _plex_number_of_points(newplex, local=True)
    meshMap = MeshMapping(newplex)
    assert Mesh(meshMap.ngMesh).GetNE(VOL) == nc

if __name__ == '__main__':
    test_ngs_plex_2d()
    test_plex_ngs_2d()
    test_ngs_plex_3d()
    test_plex_ngs_3d()
    test_plex_transform_alfeld_2d()
    test_plex_transform_alfeld_3d()
