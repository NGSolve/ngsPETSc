'''
This module contains all the functions related to wrapping NGSolve meshes to
PETSc DMPlex using the petsc4py interface.
'''
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


def buildSimplices(plex, points=None):
    """
    Return a numpy.array with the vertices of each simplex in the plex

    :arg plex: PETSc DMPlex
    :arg points: iterable of DMPlex points (must be of the same dimension)

    """
    if points is None:
        cStart, cEnd = plex.getHeightStratum(0)
        points = range(cStart, cEnd)
    vStart, vEnd = plex.getDepthStratum(0)
    T = [[v-vStart for v in plex.getAdjacency(p) if vStart <= v < vEnd] for p in points]
    return np.array(T, dtype=PETSc.IntType)


def createNetgenMesh(ngMesh, coordinates, plex, geoInfo):
    """
    Method used to generate NetgenMeshes

    :arg ngMesh: the netgen mesh to be populated
    :arg coordinates: vertices coordinates
    :arg plex: PETSc DMPlex
    :arg geoInfo: geometric information assosciated with the Netgen mesh

    """
    ngMesh.AddPoints(coordinates)
    tdim = plex.getDimension()
    gdim = plex.getCoordinateDim()
    codim = gdim - tdim

    plex.setBasicAdjacency(True, True)
    cells = buildSimplices(plex)

    surfaceLabel = FACE_SETS_LABEL if codim == 0 else CELL_SETS_LABEL
    cellIndex = plex.getLabelSize(surfaceLabel) + 1

    ngMesh.Add(ngm.FaceDescriptor(bc=1))
    ngMesh.Add(ngm.FaceDescriptor(bc=cellIndex))
    ngMesh.AddElements(dim=tdim, index=cellIndex,
                       data=cells, base=0,
                       project_geometry=geoInfo)

    fstart, fend = plex.getHeightStratum(1)
    for bcLabel in range(1, cellIndex):
        if plex.getStratumSize(FACE_SETS_LABEL, bcLabel) == 0:
            continue
        bcIndices = plex.getStratumIS(FACE_SETS_LABEL, bcLabel).indices
        fpoints = bcIndices[np.logical_and(fstart <= bcIndices, bcIndices < fend)]
        faces = buildSimplices(plex, points=fpoints)

        ngMesh.Add(ngm.FaceDescriptor(bc=bcLabel, surfnr=bcLabel))
        ngMesh.AddElements(dim=tdim-1, index=bcLabel,
                           data=faces, base=0,
                           project_geometry=geoInfo)



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
        vStart, vEnd = plex.getDepthStratum(0)
        nv = vEnd - vStart
        coordinates = plex.getCoordinatesLocal().getArray()
        gdim = plex.getCoordinateDim()
        if coordinates.size != nv * gdim:
            raise NotImplementedError("High-order mesh conversion is not supported")
        coordinates = coordinates.reshape(nv, gdim)

        self.petscPlex = plex
        ngMesh = ngm.Mesh(dim=gdim)
        self.ngMesh = ngMesh
        self.geoInfo = False
        if self.geo:
            self.ngMesh.SetGeometry(self.geo)
            self.geoInfo = True

        createNetgenMesh(self.ngMesh, coordinates, plex, self.geoInfo)

    def createPETScDMPlex(self, mesh):
        '''
        This function generates a PETSc DMPlex from a Netgen/NGSolve mesh object

        :arg mesh: the serial Netgen/NGSolve mesh object to be converted.

        '''
        if isinstance(mesh, ngs.comp.Mesh):
            self.ngMesh = mesh.ngmesh
        else:
            self.ngMesh = mesh
        if len(self.ngMesh.GetIdentifications()) > 0:
            warnings.warn("Periodic meshes are not supported by ngsPETSc" , RuntimeWarning)
        comm = self.comm
        self.geo = self.ngMesh.GetGeometry()
        els = {
            0: self.ngMesh.Elements0D,
            1: self.ngMesh.Elements1D,
            2: self.ngMesh.Elements2D,
            3: self.ngMesh.Elements3D,
        }
        gdim = self.ngMesh.dim
        tdim = gdim
        cells = els[tdim]()
        while len(cells) == 0 and tdim > 0:
            tdim -= 1
            cells = els[tdim]()
        tdim = comm.bcast(tdim, root=0)
        if comm.rank == 0:
            cells_np = cells.NumPy()
            T = trim_util(cells_np["nodes"])
            V = self.ngMesh.Coordinates()
            plex = PETSc.DMPlex().createFromCellList(tdim, T, V, comm=comm)
            vStart, _ = plex.getDepthStratum(0)
            codim_label = {0: CELL_SETS_LABEL, 1: FACE_SETS_LABEL, 2: EDGE_SETS_LABEL}
            for codim in range(tdim):
                if codim == 0 and (1 == cells_np["index"]).all():
                    continue
                for e in els[tdim - codim]():
                    join = plex.getFullJoin([vStart+v.nr-1 for v in e.vertices])
                    plex.setLabelValue(codim_label[codim], join[0], int(e.index))
        else:
            T = np.zeros((0, tdim + 1), dtype=PETSc.IntType)
            V = np.zeros((0, gdim), dtype=PETSc.RealType)
            plex = PETSc.DMPlex().createFromCellList(gdim, T, V, comm=comm)
        plex.setName(self.name)
        self.petscPlex = plex
