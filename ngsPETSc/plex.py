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
            Mesh = type("_MissingNGSolveMesh", (), {})

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

        source_type = None
        if isinstance(mesh, ngs.comp.Mesh):
            mesh = mesh.ngmesh
        if comm.rank == 0:
            if isinstance(mesh, ngm.Mesh):
                source_type = "netgen"
            elif isinstance(mesh, PETSc.DMPlex):
                source_type = "plex"
        source_type = comm.bcast(source_type, root=0)

        if source_type == "netgen":
            ngmesh = mesh
            plex = createPETScDMPlex(ngmesh, comm, name)
        elif source_type == "plex":
            plex = mesh
            ngmesh = createNetgenMesh(plex, geo)
        else:
            raise TypeError("Mesh format not recognised.")
        self.petscPlex = plex
        self.ngMesh = ngmesh
        self.comm = comm
        self.geo = (
            self.ngMesh.GetGeometry()
            if source_type == "plex" or comm.rank == 0 else None
        )
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


def addSimplices(ngMesh, dim, index, descriptor, data, project_geometry, is_occgeom):
    """
    Add simplices to a Netgen mesh

    :arg ngMesh: the Netgen Mesh
    :arg dim: the simplex dimension
    :arg index: the region index
    :arg descriptor: the region descriptor
    :arg data: a numpy.array with the vertices of each simplex
    :arg project_geometry: whether to project points to the geometry
    :arg is_occgeom: whether we have an OCCGeometry, required to decide index conventions

    """
    if descriptor is not None:
        index = ngMesh.Add(descriptor)
    elif dim == 1:
        edgenr = index-1 if is_occgeom else index
        d = ngm.EdgeDescriptor()
        d.index = index
        d.edgenr = edgenr
        index = ngMesh.Add(d)
    elif dim == 2:
        surfnr = index if is_occgeom else index-1
        index = ngMesh.Add(ngm.FaceDescriptor(bc=index, surfnr=surfnr))
    if len(data) == 0:
        return
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
    descriptors = {}
    if geo is not None:
        if isinstance(geo, ngm.Mesh):
            descriptors[1] = geo.EdgeDescriptors()
            descriptors[2] = geo.FaceDescriptors()
            geo = geo.GetGeometry()
        ngMesh.SetGeometry(geo)
        geoInfo = True
    else:
        geoInfo = False
    is_occgeom = isinstance(geo, OCCGeometry)

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
    for codim in range(tdim):
        depth = tdim - codim
        pStart, pEnd = plex.getHeightStratum(codim)

        labelName = codim_label[codim]
        labelIds = plex.getLabelIdIS(labelName).indices
        for index in sorted(labelIds):
            descr = None
            if depth in descriptors:
                descr = descriptors[depth][index-1]
            points = plex.getStratumIS(labelName, index).indices
            points = points[np.logical_and(pStart <= points, points < pEnd)]
            T = buildSimplices(plex, points=points)
            addSimplices(ngMesh, depth, index, descr, T, geoInfo, is_occgeom)

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
    addSimplices(ngMesh, tdim, index, None, T, geoInfo, is_occgeom)

    plex.setBasicAdjacency(*adjacency)
    return ngMesh


def createPETScDMPlex(ngMesh, comm, name):
    """
    Create a PETSc DMPlex from a Netgen/NGSolve mesh object

    :arg ngMesh: the serial Netgen mesh object to be converted
    :arg comm: the MPI.Comm object

    :returns: the interpolated PETSc DMPlex
    """
    if comm.rank == 0:
        els = {
            0: ngMesh.Elements0D,
            1: ngMesh.Elements1D,
            2: ngMesh.Elements2D,
            3: ngMesh.Elements3D,
        }
        if len(ngMesh.GetIdentifications()) > 0:
            warnings.warn("Periodic meshes are not supported by ngsPETSc", RuntimeWarning)
        gdim = ngMesh.dim
        tdim = gdim
        cells = els[tdim]()
        while len(cells) == 0 and tdim > 0:
            tdim -= 1
            cells = els[tdim]()
    else:
        gdim = None
        tdim = None
        cells = None
    gdim, tdim = comm.bcast((gdim, tdim), root=0)
    if comm.rank == 0:
        cells_np = cells.NumPy()
        # Netgen always stores coordinates as float64. createFromCellList performs
        # a "safe" cast to PetscReal, which rejects a float64 -> float32 narrowing
        # in single-precision PETSc builds, so cast explicitly here. This is a
        # no-op when PetscReal is double (the rank != 0 branch below already builds
        # its empty coordinate array with PETSc.RealType for the same reason).
        V = ngMesh.Coordinates().astype(PETSc.RealType, copy=False)
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
