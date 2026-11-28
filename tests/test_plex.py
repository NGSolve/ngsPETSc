'''
This module tests the plex class
'''

import numpy as np
import netgen.meshing as ngm
from mpi4py import MPI
from netgen.occ import X, OCCGeometry, WorkPlane

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
from ngsPETSc.plex import getGlobalLabelToRegionMap


def _plex_number_of_points(plex, h=0, local=False):
    points = plex.getHeightStratum(h)
    npoints = points[1] - points[0]
    if not local:
        npoints = plex.getComm().tompi4py().allreduce(npoints)
    return npoints


def _boundary_coords(coordinates, edges):
    return sorted(
        tuple(sorted(tuple(coordinates[vertex]) for vertex in edge))
        for edge in edges
    )


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
@pytest.mark.parametrize("geometric_dim", [2, 3])
def test_netgen_plex_has_region_name_labels(geometric_dim):
    """Mark the facets of each named region in a label with that name."""
    face = WorkPlane().Rectangle(1, 2).Face()
    face.edges.Min(X).name = "left"
    face.edges.Max(X).name = "right"
    ngmesh = OCCGeometry(face, dim=geometric_dim).GenerateMesh(maxh=0.5)
    plex = MeshMapping(ngmesh).petscPlex

    region_names = ngmesh.GetRegionNames(dim=1)
    for name in ("left", "right"):
        region = region_names.index(name) + 1
        assert list(plex.getLabelIdIS(name).indices) == [region]
        named_facets = plex.getStratumIS(name, region).indices
        face_sets_facets = plex.getStratumIS("Face Sets", region).indices
        assert len(named_facets) > 0
        np.testing.assert_array_equal(named_facets, face_sets_facets)


@pytest.mark.mpi_skip
def test_netgen_plex_labels_cells_of_one_named_region():
    """Mark the cells of a mesh with one named region in Cell Sets and in its label."""
    face = WorkPlane().Rectangle(1, 2).Face()
    face.faces.name = "fluid"
    ngmesh = OCCGeometry(face, dim=2).GenerateMesh(maxh=0.5)
    plex = MeshMapping(ngmesh).petscPlex

    cStart, cEnd = plex.getHeightStratum(0)
    cells = np.arange(cStart, cEnd)
    for name in ("Cell Sets", "fluid"):
        assert plex.hasLabel(name)
        np.testing.assert_array_equal(plex.getStratumIS(name, 1).indices, cells)


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

    plex.createLabel("Cell Sets")
    cStart, cEnd = plex.getHeightStratum(0)
    assert cEnd - cStart >= 1
    if comm.getSize() > 1:
        plex.setLabelValue("Cell Sets", cStart, 5 + comm.getRank())
    elif comm.getRank() == 0:
        assert cEnd - cStart >= 2
        plex.setLabelValue("Cell Sets", cStart, 5)
        plex.setLabelValue("Cell Sets", cStart + 1, 6)

    fStart, fEnd = plex.getHeightStratum(1)
    interior_faces = [face for face in range(fStart, fEnd)
                      if len(plex.getSupport(face)) == 2]
    if comm.getSize() > 1:
        assert interior_faces
        marker = 111 if comm.getRank() == 0 else 222
        plex.setLabelValue("Face Sets", interior_faces[0], marker)
    elif comm.getRank() == 0:
        assert len(interior_faces) >= 2
        plex.setLabelValue("Face Sets", interior_faces[0], 111)
        plex.setLabelValue("Face Sets", interior_faces[1], 222)

    vStart, vEnd = plex.getDepthStratum(0)
    plex_coordinates = plex.getCoordinatesLocal().getArray()
    plex_coordinates = plex_coordinates.reshape(vEnd - vStart,
                                                plex.getCoordinateDim())
    fStart, fEnd = plex.getHeightStratum(1)
    boundary_faces = [
        face for face in range(fStart, fEnd)
        if plex.getLabelValue("Face Sets", face) >= 0
    ]
    plex_boundary_edges = [
        [vertex - vStart for vertex in plex.getCone(face)]
        for face in boundary_faces
    ]
    plex_boundary_coords = _boundary_coords(plex_coordinates,
                                            plex_boundary_edges)
    label_ids, _, region_by_label = getGlobalLabelToRegionMap(
        plex, "Face Sets"
    )
    expected_region_ids = sorted(
        region_by_label[plex.getLabelValue("Face Sets", face)]
        for face in boundary_faces)

    local_gap = len(label_ids) > 0 and not np.array_equal(
        sorted(label_ids), np.arange(1, max(label_ids) + 1))
    has_gap = comm.tompi4py().allreduce(local_gap, op=MPI.LOR)
    if comm.getSize() > 1:
        assert has_gap

    ngmesh = MeshMapping(plex).ngMesh
    _, global_cell_label_ids, _ = getGlobalLabelToRegionMap(
        plex, "Cell Sets"
    )
    assert global_cell_label_ids == [5, 6]
    assert len(ngmesh.FaceDescriptors()) == len(global_cell_label_ids) + 1

    elements = ngmesh.Elements1D().NumPy()
    ng_coordinates = ngmesh.Coordinates()
    netgen_boundary_edges = [nodes[:2] - 1 for nodes in elements["nodes"]]
    netgen_boundary_coords = _boundary_coords(ng_coordinates,
                                              netgen_boundary_edges)
    actual_region_ids = sorted(np.asarray(elements["index"], dtype=int))

    np.testing.assert_allclose(netgen_boundary_coords, plex_boundary_coords)
    assert actual_region_ids == expected_region_ids


@pytest.mark.parallel([1, 2])
def test_plex_to_netgen_preserves_sparse_geometry_descriptors():
    """Keep descriptor indices when labels identify a supplied Netgen mesh."""
    comm = PETSc.COMM_WORLD
    plex = PETSc.DMPlex().createBoxMesh([2, 2], simplex=False, comm=comm)
    transform = PETSc.DMPlexTransform().create(comm=comm)
    transform.setType(PETSc.DMPlexTransformType.REFINETOSIMPLEX)
    transform.setDM(plex)
    transform.setUp()
    plex = transform.apply(plex)
    plex.distribute(overlap=0)

    # In parallel, labels 1 and 3 live on different ranks, so every rank must
    # see the global labels to agree on keeping the descriptor indices.
    fStart, fEnd = plex.getHeightStratum(1)
    interior_faces = [face for face in range(fStart, fEnd)
                      if len(plex.getSupport(face)) == 2]
    if comm.getSize() > 1:
        assert interior_faces
        local_labels = [1 if comm.getRank() == 0 else 3]
    else:
        assert len(interior_faces) >= 2
        local_labels = [1, 3]
    plex.removeLabel("Face Sets")
    plex.createLabel("Face Sets")
    for face, label in zip(interior_faces, local_labels):
        plex.setLabelValue("Face Sets", face, label)

    geo = ngm.Mesh(dim=2)
    for index in range(1, 4):
        descriptor = ngm.EdgeDescriptor()
        descriptor.index = index
        descriptor.edgenr = index
        geo.Add(descriptor)

    _, global_label_ids, _ = getGlobalLabelToRegionMap(plex, "Face Sets", 3)
    assert global_label_ids == [1, 3]

    ngmesh = MeshMapping(plex, geo=geo).ngMesh
    actual_region_ids = sorted(int(element.index)
                               for element in ngmesh.Elements1D())
    assert actual_region_ids == local_labels
    assert len(ngmesh.EdgeDescriptors()) == 3


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
