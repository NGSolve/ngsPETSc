"""Firedrake mesh/function visualization via webgui jupyter widgets.

Supports:
  * 2D triangular and 3D tetrahedral meshes.
  * Scalar and vector function spaces (Lagrange/CG, DG, RT, Nedelec, ...).
  * High polynomial orders (sub-triangulation level controllable via
    ``subdivision``; defaults to the source element degree, capped at 10).
  * Live updates via :meth:`FiredrakeScene.Redraw` for time-dependent
    simulations and :func:`ipywidgets`-driven parameter sliders.
"""

import numpy as np
from webgui_jupyter_widgets import BaseWebGuiScene, encodeData


# Families whose dofs are point evaluations: we can sample directly via FIAT
# tabulation without an intermediate interpolation step.
_POINTEVAL_FAMILIES = (
    "Lagrange", "Continuous Lagrange", "CG",
    "Discontinuous Lagrange", "DG", "DQ",
    "Q", "DPC",
)

_MAX_SUBDIVISION = 10


# ---------------------------------------------------------------------------
# Colour maps. Stops are ParaView-style ``(t, (r, g, b))`` control points with
# ``t in [0, 1]``; we resample to ``_COLORMAP_RESOLUTION`` evenly spaced RGB
# triples and hand the table to webgui via ``data["colors"]`` so it replaces
# webgui's built-in rainbow texture.
#
# ``mana`` is a 33-stop, IBM-colourblind-safe map with a pale-peach centre and
# dark navy/red ends, designed to mirror ParaView's "Cool to Warm (Extended)"
# preset whilst threading through the paper-plot palette.
# ``cool_to_warm`` is the ParaView preset of the same name.
# ``ngs`` is the original blue-cyan-green-yellow-red rainbow that NGSolve's
# webgui ships as its built-in colour map.

_MANA_STOPS = [
    (0.00000, (0.098039, 0.137255, 0.352941)),
    (0.03125, (0.207283, 0.138936, 0.373669)),
    (0.06250, (0.316527, 0.140616, 0.394398)),
    (0.09375, (0.425770, 0.142297, 0.415126)),
    (0.12500, (0.535014, 0.143978, 0.435854)),
    (0.15625, (0.644258, 0.145658, 0.456583)),
    (0.18750, (0.753501, 0.147339, 0.477311)),
    (0.21875, (0.862745, 0.149020, 0.498039)),
    (0.25000, (0.803922, 0.200490, 0.560784)),
    (0.28125, (0.745098, 0.251961, 0.623529)),
    (0.31250, (0.686275, 0.303431, 0.686275)),
    (0.34375, (0.627451, 0.354902, 0.749020)),
    (0.37500, (0.568627, 0.406373, 0.811765)),
    (0.40625, (0.509804, 0.457843, 0.874510)),
    (0.43750, (0.450980, 0.509314, 0.937255)),
    (0.46875, (0.392157, 0.560784, 1.000000)),
    (0.50000, (0.343137, 0.611765, 0.916667)),
    (0.53125, (0.294118, 0.662745, 0.833333)),
    (0.56250, (0.245098, 0.713725, 0.750000)),
    (0.59375, (0.196078, 0.764706, 0.666667)),
    (0.62500, (0.464052, 0.797386, 0.699346)),
    (0.65625, (0.732026, 0.830065, 0.732026)),
    (0.68750, (1.000000, 0.862745, 0.764706)),
    (0.71875, (0.993464, 0.790850, 0.679739)),
    (0.75000, (0.986928, 0.718954, 0.594771)),
    (0.78125, (0.980392, 0.647059, 0.509804)),
    (0.81250, (0.933333, 0.560784, 0.435294)),
    (0.84375, (0.886275, 0.474510, 0.360784)),
    (0.87500, (0.839216, 0.388235, 0.286275)),
    (0.90625, (0.729412, 0.291176, 0.245098)),
    (0.93750, (0.619608, 0.194118, 0.203922)),
    (0.96875, (0.509804, 0.097059, 0.162745)),
    (1.00000, (0.400000, 0.000000, 0.121569)),
]

_NAMED_COLORMAPS = {
    "mana": _MANA_STOPS,
}

_DEFAULT_COLORMAP = "mana"
_COLORMAP_RESOLUTION = 256


def _make_trig(N, x0=0, y0=0, dx=1, dy=1):
    return [(x0+i*dx/N,y0+j*dy/N) for j in range(N+1) for i in range(N+1-j)]

def _make_quad(N,  x0=0, y0=0, dx=1, dy=1):
    return [(x0+i*dx/N,y0+j*dy/N) for j in range(N+1) for i in range(N+1-j)] + [(x0+dx-i*dx/N,1-(y0+j*dy/N)) for j in range(N+1) for i in range(N+1-j)]

_intrules = {}
def get_intrules(dim:int, order: int):
    if (dim,order) in _intrules:
        return _intrules[(dim, order)][3] # return trig rule

    rules = {}
    if dim == 2:
        if order > 3:
            n = (order+2)//3

            trig_points = []
            h = 1/n
            for i in range(n):
                for j in range(n-i):
                    trig_points += _make_trig(3, i*h, j*h, h, h)

            for i in range(n-1):
                for j in range(n-i-1):
                    trig_points += _make_trig(3, (i+1)*h, (j+1)*h, -h, -h)

            quad_points = []
            for i in range(n):
                for j in range(n):
                    quad_points += _make_quad(3, i*h, j*h, h, h)

        else:
            trig_points =  _make_trig(order)
            quad_points =  _make_quad(order)

        rules[3] = trig_points
        rules[4] = trig_points
    elif dim == 3:
        raise RuntimeError("3D not supported")
    _intrules[(dim, order)] = rules
    return rules[3]

def _sample_colormap(spec, n=_COLORMAP_RESOLUTION):
    """Resample a ParaView-style stop list into ``n`` RGB triples.

    ``spec`` may be:
      * ``None`` — use webgui's default colormap (returns ``None``);
      * a string naming an entry in :data:`_NAMED_COLORMAPS`;
      * a list of ``(t, (r, g, b))`` tuples with ``t`` in ``[0, 1]``;
      * an ``(N, 3)`` array of RGB rows assumed to span ``[0, 1]`` evenly.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        try:
            spec = _NAMED_COLORMAPS[spec]
        except KeyError as exc:
            raise ValueError(
                f"Unknown colormap {spec!r}; choose from "
                f"{sorted(_NAMED_COLORMAPS)} or pass a stop list."
            ) from exc

    arr = np.asarray(spec, dtype=object)
    if arr.ndim == 2 and arr.shape[1] == 3 and not isinstance(spec[0][0], tuple):
        # plain (N, 3) RGB rows -> assume uniform sampling
        rgb = np.asarray(spec, dtype=np.float64)
        ts_in = np.linspace(0.0, 1.0, len(rgb))
    else:
        ts_in = np.asarray([s[0] for s in spec], dtype=np.float64)
        rgb = np.asarray([s[1] for s in spec], dtype=np.float64)

    ts_out = np.linspace(0.0, 1.0, n)
    out = np.empty((n, 3), dtype=np.float64)
    for k in range(3):
        out[:, k] = np.interp(ts_out, ts_in, rgb[:, k])
    return out.tolist()


def _get_mesh_and_func(obj, order):
    import firedrake

    func = None
    if isinstance(obj, firedrake.Function):
        mesh = obj.function_space().mesh()
        func = obj
        funcdim = 1
    elif isinstance(obj, firedrake.MeshGeometry):
        mesh = obj
        funcdim = 0
        func = firedrake.Constant(0)
    else:
        raise TypeError(f"Cannot draw object of type {type(obj)}")

    dim = mesh.geometric_dimension

    x = firedrake.SpatialCoordinate(mesh)
    if dim == 2:
        x = [x[0], x[1], 0]
        
    V = firedrake.VectorFunctionSpace(mesh, "DG", order, dim=4)
    target = firedrake.Function(V)
    func = firedrake.as_vector([*x, func])
    target.interpolate(func)

    return mesh, target, funcdim


def _value_shape(func_or_space):
    """Return the function's value shape via UFL on the bound domain."""
    if hasattr(func_or_space, "function_space"):
        fs = func_or_space.function_space()
    else:
        fs = func_or_space
    try:
        return fs.value_shape
    except AttributeError:
        pass
    try:
        return fs.ufl_element().value_shape(fs.mesh())
    except TypeError:
        return fs.ufl_element().value_shape


def _ref_lattice_tri(n):
    """Uniform lattice on UFC reference triangle.

    Returns ``(pts, tris)`` with ``pts`` of shape ``(npts, 2)`` and
    ``tris`` of shape ``(ntris, 3)``. Sub-triangles are CCW so their
    normals point in +z.
    """
    pts = np.array([[i / n, j / n] for j in range(n + 1)
                    for i in range(n + 1 - j)], dtype=np.float64)

    def idx(i, j):
        return j * (n + 1) - j * (j - 1) // 2 + i

    tris = []
    for j in range(n):
        for i in range(n - j):
            tris.append([idx(i, j), idx(i + 1, j), idx(i, j + 1)])
            if i + j < n - 1:
                tris.append([idx(i + 1, j),
                             idx(i + 1, j + 1),
                             idx(i, j + 1)])
    return pts, np.asarray(tris, dtype=np.int32)


# UFC tetrahedron facets — vertices in canonical order, opposite vertex marker.
_UFC_TET_VERTS = np.array([[0, 0, 0], [1, 0, 0],
                           [0, 1, 0], [0, 0, 1]], dtype=np.float64)
_UFC_TET_FACET_VERTS = (
    (1, 2, 3),  # facet 0 opposite V0
    (0, 2, 3),  # facet 1 opposite V1
    (0, 1, 3),  # facet 2 opposite V2
    (0, 1, 2),  # facet 3 opposite V3
)


def _facet_ref_pts_3d(local_facet, ref_pts_2d):
    """Map 2D reference triangle points to a face of the UFC tet.

    ``ref_pts_2d`` are (s, t) coordinates with s, t >= 0, s + t <= 1.
    Returns 3D ref-tet coordinates of shape ``(npts, 3)``.
    """
    f0, f1, f2 = _UFC_TET_FACET_VERTS[local_facet]
    V0 = _UFC_TET_VERTS[f0]
    V1 = _UFC_TET_VERTS[f1]
    V2 = _UFC_TET_VERTS[f2]
    s = ref_pts_2d[:, 0:1]
    t = ref_pts_2d[:, 1:2]
    return V0 + s * (V1 - V0) + t * (V2 - V0)


def _interp_for_vis(func, mesh, n):
    """Return a sample-able function and its value rank.

    For Lagrange / DG-style spaces the original ``func`` is returned.
    Non-pointwise families (RT, Nedelec, BDM, ...) are interpolated
    into a *discontinuous* DG_n / vector-DG_n space so that any genuine
    cross-cell discontinuity in the field — only the
    normal/tangential trace of an H(div)/H(curl) field is continuous —
    is preserved by the renderer.
    """
    import firedrake
    elem = func.function_space().ufl_element()
    family = elem.family()
    shape = _value_shape(func)

    V = firedrake.VectorFunctionSpace(mesh, "DG", n, dim=4)

    target = firedrake.Function(V)
    x, y = firedrake.SpatialCoordinate(mesh)
    func = firedrake.as_vector([x, y, 0, func])
    target.interpolate(func)
    return target, 1 if shape else 0


def _tabulate(func_or_coords, ref_pts):
    """FIAT-tabulate basis (order 0) of a Function/Coordinate at ref_pts.

    Returns the basis array. Scalar element shape is ``(ndof, npts)``;
    vector element shape is ``(ndof, vdim, npts)``.
    """
    fe = func_or_coords.function_space().finat_element.fiat_equivalent
    return fe.tabulate(0, ref_pts)[(0,) * ref_pts.shape[1]]


def _per_cell_eval(func, ref_pts):
    """Evaluate ``func`` at reference points within every cell.

    Returns array of shape ``(ncells, npts)`` for scalar or
    ``(ncells, npts, vdim)`` for vector functions.
    """
    # phi = _tabulate(func, ref_pts)
    # print("phi shape", phi.shape)
    print("func", func.ufl_shape)
    # print(dir(func))
    # cnm = func.cell_node_map().values
    # print('cnm1 shape', cnm.shape)
    cnm = func.at(ref_pts)
    print('cnm2 shape', len(cnm))
    print(cnm)

    

    mesh = func.function_space().mesh() 
    dim = mesh.geometric_dimension
    order = func.function_space().finat_element.degree
    ncell = mesh.num_cells()

    # physical_points = np.zeros( (ncell, ref_pts.shape[0], dim) )
    curved_points = np.zeros( (ncell, ref_pts.shape[0], dim) )

    netgen_mesh = mesh.netgen_mesh
    
    # netgen_mesh.Curve(1)
    # netgen_mesh.CalcElementMapping(ref_pts, physical_points)
    netgen_mesh.Curve(order)
    netgen_mesh.CalcElementMapping(ref_pts, curved_points)
    # curved = netgen_mesh.Elements2D().NumPy()["curved"]

    print('curved points', curved_points)
    values = np.array(func.at(curved_points.reshape(-1, dim))).reshape(ncell, ref_pts.shape[0], -1)
    print("values", values)
    return values



def _per_cell_coords(mesh, ref_pts):
    """Evaluate ``mesh.coordinates`` at reference points per cell."""
    coords = mesh.coordinates
    phi = _tabulate(coords, ref_pts)
    cnm = coords.cell_node_map().values
    dofs = coords.dat.data_ro[cnm]
    if phi.ndim == 2:
        return np.einsum('cdg,dn->cng', dofs, phi)
    return np.einsum('cd,dgn->cng', dofs, phi)


def _orient_tris_2d(verts, tris):
    """Force CCW winding so normals point +z."""
    p0 = verts[tris[:, 0]]
    p1 = verts[tris[:, 1]]
    p2 = verts[tris[:, 2]]
    cross_z = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) \
            - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    flip = cross_z < 0
    tris[flip, 1], tris[flip, 2] = tris[flip, 2].copy(), tris[flip, 1].copy()
    return tris


def _flatten_funcvals(vals):
    """Flatten per-cell, per-vertex function values to (nverts,) or (nverts, vdim)."""
    if vals.ndim == 2:
        return vals.astype(np.float32).reshape(-1)
    return vals.astype(np.float32).reshape(-1, vals.shape[-1])


def _build_2d(data, mesh, func, n, encoding):
    """Build vertex / triangle / value arrays for a 2D mesh."""
    ref_pts, ref_tris = _ref_lattice_tri(n)
    ref_pts = np.array(get_intrules(2, n))
    
    npts = ref_pts.shape[0]

    cell_coords = _per_cell_coords(mesh, ref_pts)        # (ncells, npts, gdim)
    ncells, _, gdim = cell_coords.shape

    verts3d = np.zeros((ncells * npts, 3), dtype=np.float32)
    verts3d[:, :gdim] = cell_coords.reshape(-1, gdim).astype(np.float32)

    base = (np.arange(ncells, dtype=np.int32) * npts)[:, None, None]
    tris = (base + [[0,1,2]]).reshape(-1, 3)
    
    def bernstein_triangle(n):
        from math import factorial
        def b(x, y, i, j):
            return (factorial(n) / (factorial(i) * factorial(j) * factorial(n - i - j))
                    * x**i * y**j * (1 - x - y)**(n - i - j))
    
        idx = [(i, j) for i in range(n + 1) for j in range(n + 1 - i)]
        M = np.array([[b(px / n, py / n, qx, qy) for qx, qy in idx] for px, py in idx])
        return np.linalg.inv(M)

    if func is not None:
        vals = _per_cell_eval(func, ref_pts)
        funcdim = 4
        
        ibvals = bernstein_triangle(n)
        vals = vals.reshape(-1, npts, funcdim)
        vals = vals.transpose(1, 0, 2)

        BezierPnts = np.zeros( vals.shape )
        for i in range(4):
            BezierPnts[:,:,i] = ibvals @ vals[:, :, i]

        funcmin = np.min(BezierPnts[:,:,3])
        funcmax = np.max(BezierPnts[:,:,3])

        pmin = [np.min(BezierPnts[:,:,i]) for i in range(3)]
        pmax = [np.max(BezierPnts[:,:,i]) for i in range(3)]
        center = [(pmin[i] + pmax[i]) / 2 for i in range(3)]
        radius = np.linalg.norm([pmax[i] - pmin[i] for i in range(3)]) / 2
    else:
        vals = np.zeros(ncells * npts, dtype=np.float32)
        funcdim = 0

    BezierPnts.transpose(1, 0, 2)
        
    data_2d = []    
    for i in range(npts):
        data_2d.append(encodeData(np.array(BezierPnts[i].flatten(), dtype=np.float32), np.float32, encoding))
    print('data2d')

    # data["vertices"]=  encodeData(verts3d, np.float32, self.encoding)
    data["Bezier_trig_points"]= data_2d
    data["Bezier_points"]= data_2d
    data["funcmin"] = funcmin
    data["funcmax"] = funcmax
    data["mesh_center"]= center
    data["mesh_radius"]= radius

    # Geometry edges = mesh boundary segments, sub-divided.
    segs = _build_2d_boundary_segments(mesh, n, npts, encoding)
    # return verts3d, tris, data_2d, funcdim, segs


def _build_2d_boundary_segments(mesh, n, npts, encoding):
    """Sub-segmented boundary of a 2D mesh, indexing into the per-cell lattice."""
    ef = mesh.exterior_facets
    if ef.facet_cell.size == 0:
        return encodeData(np.zeros((0, 2), dtype=np.int32), np.int32, encoding)

    facet_cells = ef.facet_cell[:, 0]
    local_facets = ef.local_facet_dat.data_ro

    # In a triangle, local facet i is the edge opposite vertex i.
    # In our lattice, vertex 0 = (0,0) -> idx 0 (j=0,i=0)
    # vertex 1 = (1,0) -> idx n
    # vertex 2 = (0,1) -> last point in column j=n -> idx (n*(n+3))//2
    def idx(i, j, m):
        return j * (m + 1) - j * (j - 1) // 2 + i

    # For each local facet, list of (i,j) pairs on the relevant edge.
    edges = {
        # opposite V0 = edge V1-V2: i+j = n
        0: [(n - j, j) for j in range(n + 1)],
        # opposite V1 = edge V0-V2: i = 0
        1: [(0, j) for j in range(n + 1)],
        # opposite V2 = edge V0-V1: j = 0
        2: [(i, 0) for i in range(n + 1)],
    }

    segs = []
    for f, cell in zip(local_facets, facet_cells):
        chain = [idx(i, j, n) + cell * npts for (i, j) in edges[int(f)]]
        for a, b in zip(chain[:-1], chain[1:]):
            segs.append((a, b))
    segs = np.asarray(segs, dtype=np.int32)
    return encodeData(segs, np.int32, encoding)


def _build_3d_boundary(mesh, func, n, encoding):
    """Build per-facet sub-triangulated boundary of a 3D tet mesh."""
    ef = mesh.exterior_facets
    if ef.facet_cell.size == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return (empty,
                np.zeros((0, 3), dtype=np.int32),
                np.zeros(0, dtype=np.float32),
                0,
                encodeData(np.zeros((0, 2), dtype=np.int32),
                           np.int32, encoding))

    ref_pts_2d, ref_tris = _ref_lattice_tri(n)
    npts = ref_pts_2d.shape[0]
    facet_cells = ef.facet_cell[:, 0]
    local_facets = ef.local_facet_dat.data_ro
    nfacets = len(local_facets)

    coords = mesh.coordinates
    coord_phi_cache = {f: _tabulate(coords, _facet_ref_pts_3d(f, ref_pts_2d))
                       for f in range(4)}
    coord_dofs = coords.dat.data_ro[coords.cell_node_map().values]  # (ncells, ndof_c, gdim)

    f_vis = func_phi_cache = func_dofs = None
    funcdim = 0
    if func is not None:
        f_vis, _ = _interp_for_vis(func, mesh, n)
        func_phi_cache = {f: _tabulate(f_vis, _facet_ref_pts_3d(f, ref_pts_2d))
                          for f in range(4)}
        func_dofs = f_vis.dat.data_ro[f_vis.cell_node_map().values]

    verts = np.empty((nfacets * npts, 3), dtype=np.float32)
    if func is not None:
        if func_dofs.ndim == 2:
            funcdim = 1
            vals = np.empty(nfacets * npts, dtype=np.float32)
        else:
            funcdim = func_dofs.shape[-1]
            vals = np.empty((nfacets * npts, funcdim), dtype=np.float32)
    else:
        vals = np.zeros(nfacets * npts, dtype=np.float32)

    tris = np.empty((nfacets * ref_tris.shape[0], 3), dtype=np.int32)
    full_coord_cnm = coords.cell_node_map().values
    parent_verts = full_coord_cnm[facet_cells]

    for k in range(nfacets):
        cell = facet_cells[k]
        f = int(local_facets[k])
        phi_c = coord_phi_cache[f]                       # (ndof_c, npts)
        c_dofs = coord_dofs[cell]                        # (ndof_c, 3)
        pts = np.einsum('dg,dn->ng', c_dofs, phi_c)      # (npts, 3)
        verts[k * npts:(k + 1) * npts] = pts.astype(np.float32)

        sub = ref_tris.copy()
        # Orient each facet's sub-triangulation so the outward normal
        # points away from the opposite tet vertex.
        v_a = pts[sub[:, 0]]
        v_b = pts[sub[:, 1]]
        v_c = pts[sub[:, 2]]
        normals = np.cross(v_b - v_a, v_c - v_a)
        opp_idx = parent_verts[k][f]
        opp_pt = mesh.coordinates.dat.data_ro[opp_idx]
        face_centroids = (v_a + v_b + v_c) / 3.0
        flip = np.sum(normals * (face_centroids - opp_pt), axis=1) < 0
        sub[flip, 1], sub[flip, 2] = sub[flip, 2].copy(), sub[flip, 1].copy()
        tris[k * ref_tris.shape[0]:(k + 1) * ref_tris.shape[0]] = sub + k * npts

        if func is not None:
            phi_f = func_phi_cache[f]
            fd_ = func_dofs[cell]
            if fd_.ndim == 1:
                vals_k = np.einsum('d,dn->n', fd_, phi_f)
            elif fd_.ndim == 2 and phi_f.ndim == 2:
                vals_k = np.einsum('dv,dn->nv', fd_, phi_f)
            else:
                vals_k = np.einsum('d,dvn->nv', fd_, phi_f)
            if vals.ndim == 1:
                vals[k * npts:(k + 1) * npts] = vals_k.astype(np.float32)
            else:
                vals[k * npts:(k + 1) * npts] = vals_k.astype(np.float32)

    # Geometry edges of the 3D boundary = edges where adjacent boundary
    # facets carry different markers. We identify these on the coarse
    # facet triangulation and then sub-divide along the lattice.
    segs = _build_3d_boundary_edges(mesh, ef, parent_verts, local_facets,
                                    n, npts, encoding)
    return verts, tris, vals, funcdim, segs


def _build_3d_boundary_edges(mesh, ef, parent_verts, local_facets,
                             n, npts, encoding):
    """Geometry edges between boundary regions with different markers,
    sub-divided into the per-facet lattice index space.
    """
    nfacets = parent_verts.shape[0]
    facet_markers = np.zeros(nfacets, dtype=np.int32)
    for m in ef.unique_markers:
        facet_markers[ef.subset(m).indices] = m

    # Coarse boundary triangle in original vertex indices.
    bnd = np.empty((nfacets, 3), dtype=np.int32)
    for k in range(nfacets):
        bnd[k] = np.delete(parent_verts[k], int(local_facets[k]))

    edge_to_facets = {}
    for k in range(nfacets):
        tri = bnd[k]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = (min(a, b), max(a, b))
            edge_to_facets.setdefault(e, []).append(k)

    # For each cross-marker edge, locate it inside each adjacent facet's
    # local lattice and emit sub-segments along the lattice.
    def lattice_idx(i, j):
        return j * (n + 1) - j * (j - 1) // 2 + i

    edge_lattice = {
        # vertex order in the local facet's _UFC_TET_FACET_VERTS:
        # local0 -> (0,0); local1 -> (n,0); local2 -> (0,n)
        (0, 1): [(i, 0) for i in range(n + 1)],   # along edge V_loc0->V_loc1
        (1, 2): [(n - j, j) for j in range(n + 1)],   # V_loc1->V_loc2
        (0, 2): [(0, j) for j in range(n + 1)],   # V_loc0->V_loc2
    }

    segs = []
    for edge, facets in edge_to_facets.items():
        markers = {facet_markers[k] for k in facets}
        if len(markers) <= 1:
            continue
        k = facets[0]
        f = int(local_facets[k])
        # Map original vertex ids -> local 0/1/2 in this facet
        face_globals = _UFC_TET_FACET_VERTS[f]
        cell = ef.facet_cell[k, 0]
        cnm = mesh.coordinates.cell_node_map().values[cell]
        local_for = {int(cnm[face_globals[0]]): 0,
                     int(cnm[face_globals[1]]): 1,
                     int(cnm[face_globals[2]]): 2}
        a, b = edge
        la, lb = local_for[a], local_for[b]
        key = (min(la, lb), max(la, lb))
        chain = edge_lattice[key]
        if (la, lb) != key:
            chain = list(reversed(chain))
        idxs = [lattice_idx(i, j) + k * npts for (i, j) in chain]
        for x, y in zip(idxs[:-1], idxs[1:]):
            segs.append((x, y))

    if not segs:
        segs = np.zeros((0, 2), dtype=np.int32)
    else:
        segs = np.asarray(segs, dtype=np.int32)
    return encodeData(segs, np.int32, encoding)


def _func_minmax(func):
    if func is None:
        return 0.0, 0.0
    vals = func.sub(3).dat.data_ro
    # if vals.ndim > 1:
        # vals = np.linalg.norm(vals, axis=1)
    return float(vals.min()), float(vals.max())


def _default_subdivision(func, mesh):
    """Pick a sensible sub-triangulation level."""
    deg = 1
    if func is not None:
        try:
            deg = int(func.function_space().ufl_element().degree())
        except (TypeError, ValueError):
            deg = 1
    coord_deg = 1
    try:
        coord_deg = int(mesh.coordinates.function_space().ufl_element().degree())
    except (TypeError, ValueError):
        coord_deg = 1
    return max(1, min(_MAX_SUBDIVISION, max(deg, coord_deg)))


class FiredrakeScene(BaseWebGuiScene):
    """A webgui scene for a Firedrake mesh or function.

    Parameters
    ----------
    obj : firedrake.MeshGeometry or firedrake.Function
        Object to visualise.
    mesh : firedrake.MeshGeometry, optional
        Override mesh (when ``obj`` is a function but you want to show
        it on a different mesh).
    subdivision : int, optional
        Sub-triangles per element edge. Defaults to the source element
        degree, capped at 10.
    colormap : str, list, ndarray, or None, optional
        Colour map used by the on-screen colour bar. Strings select a
        built-in (``"mana"`` (default), ``"cool_to_warm"``, ``"ngs"``);
        a stop list ``[(t, (r, g, b)), ...]`` or an ``(N, 3)`` RGB array
        is also accepted. Pass ``None`` to fall back to webgui's default
        rainbow (equivalent to ``"ngs"``).
    """

    def __init__(self, obj, mesh=None, subdivision=None,
                 colormap=_DEFAULT_COLORMAP, order=1, **kwargs):
        self.obj = obj
        self._mesh_override = mesh
        self.subdivision = subdivision
        self.colormap = colormap
        self.kwargs = kwargs
        self.encoding = 'b64'
        self.order = order

    def GetData(self, set_minmax=True):
        mesh, func, funcdim = _get_mesh_and_func(self.obj, self.order)
        
        tdim = mesh.topological_dimension

        d = {
            "edges": [],
            "mesh_dim": tdim,
            "funcdim": funcdim,
            "order2d": self.order,
            "order3d": self.order,
            "draw_surf": True,
            "draw_vol": False,
            "show_wireframe": False,
            "show_mesh": True,
            "is_complex": False,
            "autoscale": set_minmax,
        }

        if tdim == 2:
            _build_2d(d, mesh, func, self.order, self.encoding)
        elif tdim == 3:
            verts3d, tris, vals, funcdim, segs_enc = _build_3d_boundary(
                mesh, func, self.order, self.encoding)
        else:
            raise ValueError(f"Unsupported topological dimension {tdim}")

        if tdim == 3:
            cnm = mesh.coordinates.cell_node_map().values.astype(np.int32)
            d["tets"] = encodeData(cnm, np.int32, self.encoding)

        colors = _sample_colormap(self.colormap)
        if colors is not None:
            d["colors"] = colors

        d.update({k: v for k, v in self.kwargs.items()
                  if k in ("settings", "autoscale", "min", "max",
                           "draw_vol", "draw_surf", "show_wireframe")})
        return d

    def Redraw(self, obj=None):
        """Push fresh data to an already-displayed widget.

        Pass ``obj`` to swap to a different Function/Mesh (useful for
        slider callbacks) or call without args to re-encode the
        current object after its ``dat`` has been mutated in place
        (the time-stepping pattern).
        """
        if obj is not None:
            self.obj = obj
        super().Redraw()


def Draw(obj, mesh=None, subdivision=None, colormap=_DEFAULT_COLORMAP, order=1, **kwargs):
    """Draw a Firedrake Mesh or Function in Jupyter."""
    width = kwargs.pop("width", None)
    height = kwargs.pop("height", None)
    scene = FiredrakeScene(obj, mesh=mesh, subdivision=subdivision,
            colormap=colormap, order=order, **kwargs)
    scene.Draw(width=width, height=height)
    return scene
