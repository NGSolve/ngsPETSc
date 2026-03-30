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
try:
    from netgen.geom2d import SplineGeometry
except ImportError:
    SplineGeometry = None
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

supported_geometries = (OCCGeometry,) + ((SplineGeometry,) if SplineGeometry else ())


def projectToSplineGeo(p2d, geo):
    """
    Project a 2D point onto the nearest arc of a SplineGeometry.

    For circular arcs the centre and radius are recovered from the arc
    endpoints and their outward unit normals via:
        r = |p_start - p_end| / |n_start - n_end|
        centre = p_start - r * n_start
    For line segments the foot of the perpendicular (clamped to the
    segment) is used instead.
    """
    best_dist = np.inf
    best_proj = p2d.copy()
    for i in range(geo.GetNrLines()):
        spline = geo.GetSpline(i)
        p_start = np.asarray(spline.StartPoint())[:2]
        p_end = np.asarray(spline.EndPoint())[:2]
        n_start = np.asarray(spline.GetNormal(0))[:2]
        n_end = np.asarray(spline.GetNormal(1))[:2]
        dn = n_start - n_end
        dn_norm = np.linalg.norm(dn)
        if dn_norm < 1e-12:
            # Line segment: project to nearest point on the segment
            d = p_end - p_start
            dlen2 = float(np.dot(d, d))
            if dlen2 < 1e-30:
                p_proj = p_start.copy()
            else:
                t = float(np.clip(np.dot(p2d - p_start, d) / dlen2, 0.0, 1.0))
                p_proj = p_start + t * d
        else:
            # Circular arc
            r = np.linalg.norm(p_start - p_end) / dn_norm
            center = p_start - r * n_start
            pc = p2d - center
            pc_norm = np.linalg.norm(pc)
            p_proj = (center + r * pc / pc_norm) if pc_norm > 1e-30 else p_start.copy()
        dist = np.linalg.norm(p2d - p_proj)
        if dist < best_dist:
            best_dist = dist
            best_proj = p_proj
    return best_proj


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
    Method used to generate NetgenMeshes

    :arg plex: PETSc DMPlex
    :arg geo: Netgen geometry

    """
    # Create a Netgen Mesh
    tdim = plex.getDimension()
    gdim = plex.getCoordinateDim()
    codim = gdim - tdim
    ngMesh = ngm.Mesh(dim=gdim)
    if geo is not None:
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
    # For SplineGeometry, project boundary vertices onto the geometry arcs before
    # inserting them into the ngmesh.  Netgen's Add(edge, project_geominfo=True)
    # calls ProjectPointEdge internally, which computes the projected position but
    # never writes it back to the vertex coordinates (only epgeominfo.dist is
    # updated).  This means boundary midpoints introduced by a PETSc refinement
    # (e.g. DMPlexTransform REFINEREGULAR) stay at their linearly-interpolated
    # positions unless we correct them here.
    if geoInfo and SplineGeometry and isinstance(geo, SplineGeometry):
        fstart_tmp, fend_tmp = plex.getHeightStratum(1)
        bnd_vert_set = set()
        for lval in range(1, plex.getLabelSize(FACE_SETS_LABEL) + 1):
            if plex.getStratumSize(FACE_SETS_LABEL, lval) == 0:
                continue
            bc_ids = plex.getStratumIS(FACE_SETS_LABEL, lval).indices
            fedges = bc_ids[np.logical_and(fstart_tmp <= bc_ids, bc_ids < fend_tmp)]
            for fedge in fedges:
                for v in plex.getCone(fedge):
                    if vStart <= v < vEnd:
                        bnd_vert_set.add(v - vStart)
        if bnd_vert_set:
            coordinates = coordinates.copy()
            for vi in bnd_vert_set:
                coordinates[vi, :2] = projectToSplineGeo(coordinates[vi, :2], geo)
    ngMesh.AddPoints(coordinates)

    # Addd topology
    adjacency = plex.getBasicAdjacency()
    plex.setBasicAdjacency(True, True)
    cells = buildSimplices(plex)

    ngMesh.Add(ngm.FaceDescriptor(bc=1))
    if gdim == 2:
        cellIndex = 1
    else:
        surfaceLabel = FACE_SETS_LABEL if codim == 0 else CELL_SETS_LABEL
        cellIndex = plex.getLabelSize(surfaceLabel) + 1
        ngMesh.Add(ngm.FaceDescriptor(bc=cellIndex))

    ngMesh.AddElements(dim=tdim, index=cellIndex,
                       data=cells, base=0,
                       project_geometry=geoInfo)

    fstart, fend = plex.getHeightStratum(1)
    for bcLabel in range(1, plex.getLabelSize(FACE_SETS_LABEL) + 1):
        if plex.getStratumSize(FACE_SETS_LABEL, bcLabel) == 0:
            continue
        bcIndices = plex.getStratumIS(FACE_SETS_LABEL, bcLabel).indices
        fpoints = bcIndices[np.logical_and(fstart <= bcIndices, bcIndices < fend)]
        faces = buildSimplices(plex, points=fpoints)

        if tdim == 2:
            for face in faces:
                isSpline = SplineGeometry and isinstance(geo, SplineGeometry)
                edgenr = bcLabel if isSpline else bcLabel-1
                edge = ngm.Element1D(list(face+1), index=bcLabel, edgenr=edgenr)
                ngMesh.Add(edge, project_geominfo=geoInfo)
        else:
            ngMesh.Add(ngm.FaceDescriptor(bc=bcLabel, surfnr=bcLabel))
            ngMesh.AddElements(dim=tdim-1, index=bcLabel,
                               data=faces, base=0,
                               project_geometry=geoInfo)
    plex.setBasicAdjacency(*adjacency)
    return ngMesh


class MeshMapping:
    '''
    This class creates a mapping between Netgen/NGSolve meshes and PETSc DMPlex

    :arg mesh: the mesh object, it can be either a Netgen/NGSolve mesh or a PETSc DMPlex

    :arg name: the name of to be assigned to the PETSc DMPlex, by default this is set to "Default"

    '''

    def __init__(self, mesh=None, comm=None, geo=None, name="Default"):
        self.name = name
        if comm is None:
            comm = MPI.COMM_WORLD
        elif isinstance(comm, PETSc.Comm):
            comm = comm.tompi4py()
        self.comm = comm
        if isinstance(mesh, (ngs.comp.Mesh, ngm.Mesh)):
            self.ngMesh, self.petscPlex = self.createPETScDMPlex(mesh)
        elif isinstance(mesh, PETSc.DMPlex):
            if (geo is not None) and not isinstance(geo, supported_geometries):
                raise ValueError(
                    "Conversion from DMPlex to Netgen mesh requires OCCGeometry or SplineGeometry")
            self.ngMesh, self.petscPlex = self.createNGSMesh(mesh, geo)
        else:
            raise ValueError("Mesh format not recognised.")
        self.geo = self.ngMesh.GetGeometry()
        self.geoInfo = bool(self.geo)

    def createNGSMesh(self, plex, geo):
        '''
        This function generates an NGSolve mesh from the local part of a PETSc DMPlex

        :arg plex: the PETSc DMPlex to be converted in NGSolve mesh object
        :arg geo: Netgen geometry

        '''
        ngMesh = createNetgenMesh(plex, geo)
        return ngMesh, plex

    def createPETScDMPlex(self, mesh):
        '''
        This function generates a PETSc DMPlex from a Netgen/NGSolve mesh object

        :arg mesh: the serial Netgen/NGSolve mesh object to be converted.

        '''
        if isinstance(mesh, ngs.comp.Mesh):
            ngMesh = mesh.ngmesh
        else:
            ngMesh = mesh
        if len(ngMesh.GetIdentifications()) > 0:
            warnings.warn("Periodic meshes are not supported by ngsPETSc" , RuntimeWarning)
        comm = self.comm
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
            T = trim_util(cells_np["nodes"])
            V = ngMesh.Coordinates()
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
        plex.setName(self.name)
        return ngMesh, plex
