'''
This module test that plex class
'''
import pytest

from ngsolve import Mesh, VOL
from netgen.geom2d import unit_square
from netgen.csg import unit_cube

from petsc4py import PETSc

from ngsPETSc import MeshMapping

@pytest.mark.mpi_skip()
def test_ngs_plex_2d():
    '''
    Testing the conversion from NGSolve mesh to PETSc DMPlex
    for a two dimensional simplex mesh
    '''
    mesh = Mesh(unit_square.GenerateMesh(maxh=1.))
    meshMap = MeshMapping(mesh)
    assert meshMap.petscPlex.getHeightStratum(0)[1] == 2

@pytest.mark.mpi_skip()
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
    meshMap = MeshMapping(plex)
    assert Mesh(meshMap.ngsMesh).GetNE(VOL) == 8

@pytest.mark.mpi_skip()
def test_ngs_plex_3d():
    '''
    Testing the conversion from NGSolve mesh to PETSc DMPlex
    for a three dimensional simplex mesh
    '''
    mesh = Mesh(unit_cube.GenerateMesh(maxh=1.))
    meshMap = MeshMapping(mesh)
    assert meshMap.petscPlex.getHeightStratum(0)[1] == 12

@pytest.mark.mpi_skip()
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
    meshMap = MeshMapping(plex)
    assert Mesh(meshMap.ngsMesh).GetNE(VOL) == 6

@pytest.mark.mpi_skip()
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
    meshMap = MeshMapping(newplex)
    assert Mesh(meshMap.ngsMesh).GetNE(VOL) == 6

@pytest.mark.mpi_skip()
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
    meshMap = MeshMapping(newplex)
    assert Mesh(meshMap.ngsMesh).GetNE(VOL) == 48

@pytest.mark.mpi_skip()
def test_plex_transform_box_2d():
    '''
    Testing the use of the PETSc Alfeld transform 
    on a NGSolve mesh.
    '''
    mesh = Mesh(unit_square.GenerateMesh(maxh=1.))
    meshMap = MeshMapping(mesh)
    tr = PETSc.DMPlexTransform().create(comm=PETSc.COMM_WORLD)
    tr.setType(PETSc.DMPlexTransformType.REFINETOBOX)
    tr.setDM(meshMap.petscPlex)
    tr.setUp()
    newplex = tr.apply(meshMap.petscPlex)
    meshMap = MeshMapping(newplex)
    assert Mesh(meshMap.ngsMesh).GetNE(VOL) == 6
