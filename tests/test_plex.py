'''
This module test the plex class
'''

from ngsolve import Mesh, VOL
from netgen.geom2d import unit_square
from netgen.csg import unit_cube

from petsc4py import PETSc

from ngsPETSc import MeshMapping

def _plex_number_of_points(plex, h=0, local=False):
    points = plex.getHeightStratum(h)
    np = points[1] - points[0]
    if not local:
        np = plex.getComm().tompi4py().allreduce(np)
    return np

def test_ngs_plex_2d():
    '''
    Testing the conversion from NGSolve mesh to PETSc DMPlex
    for a two dimensional simplex mesh
    '''
    mesh = Mesh(unit_square.GenerateMesh(maxh=1.))
    meshMap = MeshMapping(mesh)
    plex = meshMap.petscPlex
    assert _plex_number_of_points(plex) == 2

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
    assert Mesh(meshMap.ngMesh).GetNE(VOL) == nc

def test_ngs_plex_3d():
    '''
    Testing the conversion from NGSolve mesh to PETSc DMPlex
    for a three dimensional simplex mesh
    '''
    mesh = Mesh(unit_cube.GenerateMesh(maxh=1.))
    meshMap = MeshMapping(mesh)
    plex = meshMap.petscPlex
    assert _plex_number_of_points(plex) == 12

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
