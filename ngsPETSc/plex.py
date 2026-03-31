'''
This module contains all the functions related to wrapping NGSolve meshes to
PETSc DMPlex using the petsc4py interface.
'''
import warnings
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
import netgen.meshing as ngm
from netgen.occ import OCCGeometry
from .utils.utils import trim_util
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


class MeshMapping:
    """
    A mapping between a Netgen/NGSolve mesh and a PETSc DMPlex

    :arg mesh: the source mesh, either a Netgen/NGSolve mesh or a PETSc DMPlex
    :kwarg comm: an optional MPI.Comm
    :kwarg geo: the underlying Netgen geometry, ignored if mesh is a Netgen mesh
    :kwarg name: the name of to be assigned to the PETSc DMPlex, by default this is set to "Default"
    """
    def __init__(self, mesh, comm=None, geo=None, name="Default"):
        if comm is None:
            comm = MPI.COMM_WORLD
        elif isinstance(comm, PETSc.Comm):
            comm = comm.tompi4py()

        if isinstance(mesh, ngs.comp.Mesh):
            mesh = mesh.ngmesh

        if isinstance(mesh, ngm.Mesh):
            ngmesh = mesh
            plex = createPETScDMPlex(ngmesh, comm, name)
        elif isinstance(mesh, PETSc.DMPlex):
            plex = mesh
            ngmesh = createNetgenMesh(plex, geo)
        else:
            raise TypeError("Mesh format not recognised.")
        self.petscPlex = plex
        self.ngMesh = ngmesh
        self.comm = comm
        self.geo = self.ngMesh.GetGeometry()
        self.geoInfo = bool(self.geo)


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


def createNetgenMesh(plex, geo):
    """
    Create a Netgen mesh from the local part of a PETSc DMPlex

    :arg plex: the PETSc DMPlex to be converted in NGSolve mesh object
    :arg geo: Netgen geometry or Netgen mesh to extract geometry from

    """
    # Create a Netgen Mesh
    tdim = plex.getDimension()
    gdim = plex.getCoordinateDim()
    ngMesh = ngm.Mesh(dim=gdim)
    edgenr_mapping = None
    if geo is not None:
        if isinstance(geo, ngm.Mesh):
            edgenr_mapping = {e.index: e.edgenr for e in geo.Elements1D()}
            geo = geo.GetGeometry()
        ngMesh.SetGeometry(geo)
        geoInfo = True
    else:
        geoInfo = False

    # Add vertices
    vStart, vEnd = plex.getDepthStratum(0)
    nv = vEnd - vStart
    coordinates = plex.getCoordinatesLocal().getArray()
    if coordinates.size != nv * gdim:
        raise NotImplementedError("High-order mesh conversion is not supported")
    coordinates = coordinates.reshape(nv, gdim)
    ngMesh.AddPoints(coordinates)

    # Set adjacency
    adjacency = plex.getBasicAdjacency()
    plex.setBasicAdjacency(True, True)

    # Add labeled entities
    codim_label = {0: CELL_SETS_LABEL, 1: FACE_SETS_LABEL, 2: EDGE_SETS_LABEL}
    for depth in range(1, tdim+1):
        codim = tdim - depth
        pStart, pEnd = plex.getHeightStratum(codim)

        labelName = codim_label[codim]
        labelIds = plex.getLabelIdIS(labelName).indices
        for index in labelIds:
            if plex.getStratumSize(labelName, index) == 0:
                continue

            points = plex.getStratumIS(labelName, index).indices
            points = points[np.logical_and(pStart <= points, points < pEnd)]
            T = buildSimplices(plex, points=points)
            if depth == 1:
                if edgenr_mapping is not None:
                    edgenr = edgenr_mapping[index]
                else:
                    edgenr = index-1 if isinstance(geo, OCCGeometry) else index
                T += 1
                for Te in T:
                    edge = ngm.Element1D(list(Te), index=index, edgenr=edgenr)
                    ngMesh.Add(edge, project_geominfo=geoInfo)
            else:
                if depth == 2:
                    surfnr = index if isinstance(geo, OCCGeometry) else index-1
                    index = ngMesh.Add(ngm.FaceDescriptor(bc=index, surfnr=surfnr))
                ngMesh.AddElements(dim=depth, index=index, data=T, base=0,
                                   project_geometry=geoInfo)

    # Add unlabeled cells
    labelName = codim_label[0]
    if plex.getLabelSize(labelName) > 0:
        cStart, cEnd = plex.getHeightStratum(0)
        labelIds = plex.getLabelIdIS(labelName).indices
        points = np.concatenate([plex.getStratumIS(labelName, index).indices for index in labelIds])
        points = np.setdiff1d(np.arange(cStart, cEnd), points)
    else:
        points = None
    cells = buildSimplices(plex, points=points)
    index = plex.getLabelSize(labelName) + 1
    if tdim == 2:
        surfnr = index if isinstance(geo, OCCGeometry) else index-1
        index = ngMesh.Add(ngm.FaceDescriptor(bc=index, surfnr=surfnr))
    ngMesh.AddElements(dim=tdim, index=index,
                       data=cells, base=0,
                       project_geometry=geoInfo)

    plex.setBasicAdjacency(*adjacency)
    return ngMesh


def createPETScDMPlex(ngMesh, comm, name):
    """
    Create a PETSc DMPlex from a Netgen/NGSolve mesh object

    :arg ngMesh: the serial Netgen mesh object to be converted
    :arg comm: the MPI.Comm object

    :returns: a tuple of Netgen mesh and DMPlex
    """
    if len(ngMesh.GetIdentifications()) > 0:
        warnings.warn("Periodic meshes are not supported by ngsPETSc" , RuntimeWarning)
    els = {
        0: ngMesh.Elements0D,
        1: ngMesh.Elements1D,
        2: ngMesh.Elements2D,
        3: ngMesh.Elements3D,
    }
    gdim = ngMesh.dim
    tdim = gdim
    cells = els[tdim]()
    while len(cells) == 0 and tdim > 0:
        tdim -= 1
        cells = els[tdim]()
    tdim = comm.bcast(tdim, root=0)
    if comm.rank == 0:
        cells_np = cells.NumPy()
        V = ngMesh.Coordinates()
        T = trim_util(cells_np["nodes"])
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
        T = np.empty((0, tdim + 1), dtype=PETSc.IntType)
        V = np.empty((0, gdim), dtype=PETSc.RealType)
        plex = PETSc.DMPlex().createFromCellList(tdim, T, V, comm=comm)
    plex.setName(name)
    return plex
