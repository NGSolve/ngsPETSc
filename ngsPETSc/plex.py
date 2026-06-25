'''
This module contains all the functions related to wrapping NGSolve meshes to
PETSc DMPlex using the petsc4py interface.
'''
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
    if len(data) == 0:
        return
    if dim == 1:
        edgenr = index-1 if isoccgeom else index
        if edgenr_mapping is not None:
            edgenr = edgenr_mapping.get(index, edgenr)
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


def buildPeriodicVertexMap(ngMesh):
    """
    Build the vertex renumbering that collapses periodically identified vertices.

    Netgen stores a periodic mesh as an ordinary (non-periodic) mesh of the domain
    plus a list of identification pairs ``(p1, p2)`` of vertices on opposite
    periodic boundaries. To obtain a topologically periodic DMPlex, every set of
    identified vertices must be merged into a single representative vertex. All
    identifications are treated as periodic (Netgen's Python ``Mesh`` does not
    expose the identification type).

    :arg ngMesh: the serial Netgen mesh object

    :returns: a tuple ``(old_to_new, survivors, identified)`` where
        ``old_to_new`` maps each 0-based Netgen vertex index to a dense 0-based
        index in the merged vertex numbering, ``survivors`` is the sorted array of
        original 0-based indices of the surviving (representative) vertices, and
        ``identified`` is the set of 0-based vertex indices that participate in an
        identification (used to drop the periodic seam from the boundary labels).
    """
    nv = len(ngMesh.Coordinates())
    # Union-find over the identified vertices, keeping the smallest index in each
    # group as its representative for a deterministic numbering.
    parent = np.arange(nv, dtype=np.int64)

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    identified = set()
    for p1, p2 in ngMesh.GetIdentifications():
        # GetIdentifications returns pairs of PointId objects whose .nr is 1-based.
        a, b = p1.nr - 1, p2.nr - 1
        identified.add(a)
        identified.add(b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    roots = np.array([find(i) for i in range(nv)], dtype=np.int64)
    survivors = np.unique(roots)
    compact = np.empty(nv, dtype=PETSc.IntType)
    compact[survivors] = np.arange(survivors.size, dtype=PETSc.IntType)
    old_to_new = compact[roots]
    return old_to_new, survivors, identified


def createPETScDMPlex(ngMesh, comm, name):
    """
    Create a PETSc DMPlex from a Netgen/NGSolve mesh object

    Periodic Netgen meshes are supported: vertices that Netgen identifies across
    periodic boundaries are merged so that the resulting DMPlex is topologically
    periodic. The geometry is left "wrapped" at this stage (only the representative
    vertex coordinates are stored); a discontinuous coordinate field carrying the
    un-wrapped per-cell coordinates is attached downstream (e.g. by Firedrake).

    :arg ngMesh: the serial Netgen mesh object to be converted
    :arg comm: the MPI.Comm object

    :returns: a tuple of Netgen mesh and DMPlex
    """
    periodic = len(ngMesh.GetIdentifications()) > 0
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
        # Netgen always stores coordinates as float64. createFromCellList performs
        # a "safe" cast to PetscReal, which rejects a float64 -> float32 narrowing
        # in single-precision PETSc builds, so cast explicitly here. This is a
        # no-op when PetscReal is double (the rank != 0 branch below already builds
        # its empty coordinate array with PETSc.RealType for the same reason).
        V = ngMesh.Coordinates().astype(PETSc.RealType, copy=False)
        T = trim_util(cells_np["nodes"])
        if periodic:
            # Merge periodically identified vertices to obtain a periodic topology.
            old_to_new, survivors, identified = buildPeriodicVertexMap(ngMesh)
            V = V[survivors]
            T = old_to_new[T]
            # A cell that contains an identified vertex pair spans a full period and
            # collapses to a degenerate cell once the vertices are merged. This only
            # happens when the mesh is too coarse along the periodic direction.
            degenerate = np.array([len(np.unique(row)) < T.shape[1] for row in T])
            if degenerate.any():
                raise ValueError(
                    f"{int(degenerate.sum())} cell(s) span a full period and become "
                    "degenerate when the periodic vertices are merged. Refine the "
                    "mesh along the periodic direction(s).")
        plex = PETSc.DMPlex().createFromCellList(tdim, T, V, comm=comm)
        vStart, _ = plex.getDepthStratum(0)
        codim_label = {0: CELL_SETS_LABEL, 1: FACE_SETS_LABEL, 2: EDGE_SETS_LABEL}
        for codim in range(tdim):
            if codim == 0 and (1 == cells_np["index"]).all():
                continue
            for e in els[tdim - codim]():
                vnums = [v.nr-1 for v in e.vertices]
                if periodic and codim > 0:
                    # A boundary entity all of whose vertices are identified lies on
                    # the periodic seam: after merging it becomes interior, so it must
                    # not be labelled as a boundary (mirrors Firedrake's periodic
                    # meshes, which leave the periodic boundary ids empty).
                    if all(vn in identified for vn in vnums):
                        continue
                    vnums = [old_to_new[vn] for vn in vnums]
                join = plex.getFullJoin([vStart+vn for vn in vnums])
                plex.setLabelValue(codim_label[codim], join[0], int(e.index))
    else:
        T = np.empty((0, tdim + 1), dtype=PETSc.IntType)
        V = np.empty((0, gdim), dtype=PETSc.RealType)
        plex = PETSc.DMPlex().createFromCellList(tdim, T, V, comm=comm)
    plex.setName(name)
    return plex
