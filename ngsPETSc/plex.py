'''
This module contains all the functions related to wrapping NGSolve meshes to
PETSc DMPlex using the petsc4py interface.
'''
from collections import defaultdict
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


def addSimplices(ngMesh, dim, index, data, project_geometry, isoccgeom, edgenr_mapping):
    """
    Add simplices to a Netgen mesh

    :arg ngMesh: the Netgen Mesh
    :arg dim: the simplex dimension
    :arg index: the region index
    :arg data: a numpy.array with the vertices of each simplex
    :project_geometry: whether to project points to the geometry
    :isoccgeom: whether we have an OCCGeometry, required to decide index conventions
    :edgenr_mapping: a dict mapping from region index to edgenr

    """
    if dim == 1:
        if edgenr_mapping is not None:
            edgenr = edgenr_mapping[index]
        else:
            edgenr = index-1 if isoccgeom else index
        for edge in data:
            ngMesh.Add(ngm.Element1D(list(edge+1), index=index, edgenr=edgenr),
                       project_geominfo=project_geometry)
    else:
        if dim == 2:
            surfnr = index if isoccgeom else index-1
            index = ngMesh.Add(ngm.FaceDescriptor(bc=index, surfnr=surfnr))
        ngMesh.AddElements(dim=dim, index=index, data=data, base=0,
                           project_geometry=project_geometry)


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
    isoccgeom = isinstance(geo, OCCGeometry)

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
            addSimplices(ngMesh, depth, index, T, geoInfo, isoccgeom, edgenr_mapping)

    # Add unlabeled cells
    labelName = codim_label[0]
    if plex.getLabelSize(labelName) > 0:
        cStart, cEnd = plex.getHeightStratum(0)
        labelIds = plex.getLabelIdIS(labelName).indices
        points = np.concatenate([plex.getStratumIS(labelName, index).indices for index in labelIds])
        points = np.setdiff1d(np.arange(cStart, cEnd), points)
    else:
        points = None
    index = plex.getLabelSize(labelName) + 1
    T = buildSimplices(plex, points=points)
    addSimplices(ngMesh, tdim, index, T, geoInfo, isoccgeom, edgenr_mapping)

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

    codim_entity = {0: "cell", 1: "face", 2: "edge"}
    for codim in range(tdim):
        label_mapping = defaultdict(list)
        if comm.rank == 0:
            regions = ngMesh.GetRegionNames(codim=codim)
            for label, rname in enumerate(regions, start=1):
                label_mapping[rname].append(label)

        label_mapping = comm.bcast(dict(label_mapping), root=0)
        entity = codim_entity[codim]
        plex.setAttr(f"{entity}_region_names", label_mapping)

    return plex
