'''
This module contains all the functions related to wrapping NGSolve meshes to
PETSc DMPlex using the petsc4py interface.
'''
import itertools
import warnings
import numpy as np
from petsc4py import PETSc
import netgen.meshing as ngm
from ngsPETSc.utils.utils import trim_util
try:
    import ngsolve as ngs
except ImportError:
    class ngs:
        "dummy class"
        class comp:
            "dummy class"
            Mesh = type(None)

FACE_SETS_LABEL = "Face Sets"
CELL_SETS_LABEL = "Cell Sets"
EDGE_SETS_LABEL = "Edge Sets"


def buildCells(coordinates, plex):
    """
    Return a numpy.array with the vertices of each cell

    :arg coordinates: vertices coordinates
    :arg plex: PETSc DMPlex

    """
    cStart, cEnd = plex.getHeightStratum(0)
    vStart, vEnd = plex.getDepthStratum(0)

    cells = [[v-vStart for v in plex.getAdjacency(c) if vStart <= v < vEnd]
             for c in range(cStart, cEnd)]
    cells = np.array(cells)
    return cells


def create2DNetgenMesh(ngMesh, coordinates, plex, geoInfo):
    """
    Method used to generate 2D NetgenMeshes

    :arg ngMesh: the netgen mesh to be populated
    :arg coordinates: vertices coordinates
    :arg plex: PETSc DMPlex
    :arg geoInfo: geometric information assosciated with the Netgen mesh

    """
    ngMesh.AddPoints(coordinates)
    cells = buildCells(coordinates, plex)
    vStart, _ = plex.getDepthStratum(0)

    ngMesh.Add(ngm.FaceDescriptor(bc=1))
    if cells.ndim == 2:
        ngMesh.AddElements(dim=2, index=1, data=cells, base=0)
    for bcLabel in range(1, plex.getLabelSize(FACE_SETS_LABEL)+1):
        if plex.getStratumSize("Face Sets", bcLabel) == 0:
            continue
        bcIndices = plex.getStratumIS("Face Sets", bcLabel).indices
        for j in bcIndices:
            bcIndex = plex.getCone(j) - vStart + 1
            if len(bcIndex) == 2:
                edge = ngm.Element1D(list(bcIndex),
                                     index=bcLabel,
                                     edgenr=bcLabel-1)
                ngMesh.Add(edge, project_geominfo=geoInfo)


def create3DNetgenMesh(ngMesh, coordinates, plex, geoInfo):
    """
    Method used to generate 3D NetgenMeshes

    :arg ngMesh: the netgen mesh to be populated
    :arg coordinates: vertices coordinates
    :arg plex: PETSc DMPlex
    :arg geoInfo: geometric information assosciated with the Netgen mesh

    """
    ngMesh.AddPoints(coordinates)
    cells = buildCells(coordinates, plex)
    vStart, _ = plex.getDepthStratum(0)

    ngMesh.Add(ngm.FaceDescriptor(bc=1))
    ngMesh.Add(ngm.FaceDescriptor(bc=plex.getLabelSize(FACE_SETS_LABEL)+1))
    if cells.ndim == 2:
        ngMesh.AddElements(dim=3, index=plex.getLabelSize(FACE_SETS_LABEL)+1,
                           data=cells, base=0)
    for bcLabel in range(1, plex.getLabelSize(FACE_SETS_LABEL)+1):
        faces = []
        if plex.getStratumSize("Face Sets", bcLabel) == 0:
            continue
        bcIndices = plex.getStratumIS("Face Sets", bcLabel).indices
        for j in bcIndices:
            sIndex  = plex.getCone(j)
            if len(sIndex) == 3:
                S = dict.fromkeys(itertools.chain.from_iterable(
                    plex.getCone(p) - vStart
                    for p in sIndex)
                )
                face = list(S)
                A = coordinates[face][[1, 2, 0]] - coordinates[face]
                eig = np.linalg.eig(A)[0].real
                if eig[1] * eig[2] < 0:
                    face = [face[0], face[2], face[1]]
                faces.append(face)

        ngMesh.Add(ngm.FaceDescriptor(bc=bcLabel, surfnr=bcLabel))
        ngMesh.AddElements(dim=2, index=bcLabel,
                           data=np.asarray(faces, dtype=np.int32), base=0,
                           project_geometry = geoInfo)


class MeshMapping:
    '''
    This class creates a mapping between Netgen/NGSolve meshes and PETSc DMPlex

    :arg mesh: the mesh object, it can be either a Netgen/NGSolve mesh or a PETSc DMPlex

    :arg name: the name of to be assigned to the PETSc DMPlex, by default this is set to "Default"

    '''

    def __init__(self, mesh=None, comm=None, geo=None, name="Default"):
        self.name = name
        self.comm = comm if comm is not None else PETSc.COMM_WORLD
        self.geo = geo
        if isinstance(mesh,(ngs.comp.Mesh,ngm.Mesh)):
            self.createPETScDMPlex(mesh)
        elif isinstance(mesh,PETSc.DMPlex):
            self.createNGSMesh(mesh)
        else:
            raise ValueError("Mesh format not recognised.")

    def createNGSMesh(self, plex):
        '''
        This function generates an NGSolve mesh from the local part of a PETSc DMPlex

        :arg plex: the PETSc DMPlex to be converted in NGSolve mesh object

        '''
        if plex.getDimension() not in [2,3]:
            raise NotImplementedError(f"Not implemented for dimension {plex.getDimension()}.")
        nv = plex.getDepthStratum(0)[1] - plex.getDepthStratum(0)[0]
        coordinates = plex.getCoordinatesLocal().getArray()
        if coordinates.shape[0] != nv * plex.getDimension():
            raise NotImplementedError("High-order mesh conversion is not supported")

        self.petscPlex = plex
        ngMesh = ngm.Mesh(dim=plex.getCoordinateDim())
        self.ngMesh = ngMesh
        self.geoInfo = False
        if self.geo:
            self.ngMesh.SetGeometry(self.geo)
            self.geoInfo = True

        coordinates = coordinates.reshape(-1,plex.getDimension())
        if plex.getDimension() == 2:
            create2DNetgenMesh(self.ngMesh, coordinates, plex, self.geoInfo)
        elif plex.getDimension() == 3:
            create3DNetgenMesh(self.ngMesh, coordinates, plex, self.geoInfo)
        else:
            raise NotImplementedError("No implementation for dimension greater than 3.")

    def createPETScDMPlex(self, mesh):
        '''
        This function generate an PETSc DMPlex from a Netgen/NGSolve mesh object

        :arg mesh: the serial Netgen/NGSolve mesh object to be converted.

        '''
        if isinstance(mesh,ngs.comp.Mesh):
            self.ngMesh = mesh.ngmesh
        else:
            self.ngMesh = mesh
        if len(self.ngMesh.GetIdentifications()) > 0:
            warnings.warn("Periodic meshes are not supported by ngsPETSc" , RuntimeWarning)
        comm = self.comm
        self.geo = self.ngMesh.GetGeometry()
        if self.ngMesh.dim == 3:
            if comm.rank == 0:
                V = self.ngMesh.Coordinates()
                T = self.ngMesh.Elements3D().NumPy()["nodes"]

                surfMesh, dim = False, 3
                if len(T) == 0:
                    surfMesh, dim = True, 2
                    T = self.ngMesh.Elements2D().NumPy()["nodes"]

                T  = trim_util(T)

                plex = PETSc.DMPlex().createFromCellList(dim, T, V, comm=comm)
                plex.setName(self.name)
                vStart, _ = plex.getDepthStratum(0)
                if surfMesh:
                    for e in self.ngMesh.Elements1D():
                        join = plex.getJoin([vStart+v.nr-1 for v in e.vertices])
                        plex.setLabelValue(FACE_SETS_LABEL, join[0], int(e.surfaces[1]))
                else:
                    for e in self.ngMesh.Elements2D():
                        join = plex.getFullJoin([vStart+v.nr-1 for v in e.vertices])
                        plex.setLabelValue(FACE_SETS_LABEL, join[0], int(e.index))
                    for e in self.ngMesh.Elements1D():
                        join = plex.getJoin([vStart+v.nr-1 for v in e.vertices])
                        plex.setLabelValue(EDGE_SETS_LABEL, join[0], int(e.index))
                self.petscPlex = plex
            else:
                plex = PETSc.DMPlex().createFromCellList(3,
                                                        np.zeros((0, 4), dtype=np.int32),
                                                        np.zeros((0, 3), dtype=np.double),
                                                        comm=comm)
                self.petscPlex = plex
        elif self.ngMesh.dim == 2:
            if comm.rank == 0:
                V = self.ngMesh.Coordinates()
                T = self.ngMesh.Elements2D().NumPy()["nodes"]
                T = np.array([list(np.trim_zeros(a, 'b')) for a in list(T)])-1
                plex = PETSc.DMPlex().createFromCellList(2, T, V, comm=comm)
                plex.setName(self.name)
                vStart, _ = plex.getDepthStratum(0)   # vertices
                for e in self.ngMesh.Elements1D():
                    join = plex.getJoin([vStart+v.nr-1 for v in e.vertices])
                    plex.setLabelValue(FACE_SETS_LABEL, join[0], int(e.index))
                if not (1 == self.ngMesh.Elements2D().NumPy()["index"]).all():
                    for e in self.ngMesh.Elements2D():
                        join = plex.getFullJoin([vStart+v.nr-1 for v in e.vertices])
                        plex.setLabelValue(CELL_SETS_LABEL, join[0], int(e.index))

                self.petscPlex = plex
            else:
                plex = PETSc.DMPlex().createFromCellList(2,
                                                        np.zeros((0, 3), dtype=np.int32),
                                                        np.zeros((0, 2), dtype=np.double),
                                                        comm=comm)
                self.petscPlex = plex
