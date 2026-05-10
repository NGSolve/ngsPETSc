"""Firedrake mesh/function visualization via webgui jupyter widgets.

Mirrors the NGSolve ``BuildRenderData`` scheme so webgui's WebGL shaders
receive the data layout they expect:

* Surfaces (2D triangles and 3D boundary facets) are sent as cubic
  Bernstein/Bézier control points; source elements of degree > 3 are
  sub-triangulated into multiple cubic patches.
* 3D meshes with a function additionally ship ``points3d`` so the
  clipping shader can slice the volume.

Supports scalar and vector function spaces (Lagrange/CG, DG, RT,
Nedelec, BDM), live updates via :meth:`FiredrakeScene.Redraw`, and
ParaView-style colour maps.
"""

import math
from functools import lru_cache

import numpy as np
from webgui_jupyter_widgets import BaseWebGuiScene, encodeData


# Families whose dofs are point evaluations: we can sample directly via FIAT
# tabulation without an intermediate interpolation step.
_POINTEVAL_FAMILIES = (
    "Lagrange", "Continuous Lagrange", "CG",
    "Discontinuous Lagrange", "DG", "DQ",
    "Q", "DPC",
)

# Webgui's WebGL shaders cap at cubic Bernstein on triangles and P2 on tets;
# sub-triangulation handles higher-degree sources on the surface.
_MAX_ORDER_2D = 3
_MAX_ORDER_3D = 2
_MAX_SOURCE_DEGREE = 10


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

_COOL_TO_WARM_STOPS = [
    (0.00000, (0.000000, 0.000000, 0.349020)),
    (0.03125, (0.039216, 0.062745, 0.380392)),
    (0.06250, (0.062745, 0.117647, 0.411765)),
    (0.09375, (0.090196, 0.184314, 0.450980)),
    (0.12500, (0.125490, 0.262745, 0.501961)),
    (0.15625, (0.160784, 0.337255, 0.541176)),
    (0.18750, (0.200000, 0.396078, 0.568627)),
    (0.21875, (0.239216, 0.454902, 0.600000)),
    (0.25000, (0.286275, 0.521569, 0.650980)),
    (0.28125, (0.337255, 0.592157, 0.701961)),
    (0.31250, (0.388235, 0.654902, 0.749020)),
    (0.34375, (0.466667, 0.737255, 0.819608)),
    (0.37500, (0.572549, 0.819608, 0.878431)),
    (0.40625, (0.654902, 0.866667, 0.909804)),
    (0.43750, (0.752941, 0.917647, 0.941176)),
    (0.46875, (0.823529, 0.956863, 0.968627)),
    (0.50000, (0.988235, 0.960784, 0.901961)),
    (0.51562, (0.941176, 0.984314, 0.988235)),
    (0.53125, (0.988235, 0.945098, 0.850980)),
    (0.56250, (0.980392, 0.898039, 0.784314)),
    (0.59375, (0.968627, 0.835294, 0.698039)),
    (0.62500, (0.949020, 0.733333, 0.588235)),
    (0.65625, (0.929412, 0.650980, 0.509804)),
    (0.68750, (0.909804, 0.564706, 0.435294)),
    (0.71875, (0.878431, 0.458824, 0.352941)),
    (0.75000, (0.839216, 0.388235, 0.286275)),
    (0.78125, (0.760784, 0.294118, 0.211765)),
    (0.81250, (0.701961, 0.211765, 0.168627)),
    (0.84375, (0.650980, 0.156863, 0.129412)),
    (0.87500, (0.600000, 0.094118, 0.094118)),
    (0.90625, (0.549020, 0.066667, 0.098039)),
    (0.93750, (0.501961, 0.050980, 0.125490)),
    (0.96875, (0.450000, 0.054902, 0.172549)),
    (1.00000, (0.400000, 0.000000, 0.121569)),
]

# NGSolve webgui's built-in rainbow: piecewise-linear blue-cyan-green-yellow-red.
_NGS_STOPS = [
    (0.00, (0.0, 0.0, 1.0)),
    (0.25, (0.0, 1.0, 1.0)),
    (0.50, (0.0, 1.0, 0.0)),
    (0.75, (1.0, 1.0, 0.0)),
    (1.00, (1.0, 0.0, 0.0)),
]

_NAMED_COLORMAPS = {
    "mana": _MANA_STOPS,
    "cool_to_warm": _COOL_TO_WARM_STOPS,
    "ngs": _NGS_STOPS,
}

_DEFAULT_COLORMAP = "mana"
_COLORMAP_RESOLUTION = 256


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


# ---------------------------------------------------------------------------
# Mesh / function plumbing
# ---------------------------------------------------------------------------

def _get_mesh_and_func(obj):
    import firedrake
    if isinstance(obj, firedrake.Function):
        return obj.function_space().mesh(), obj
    if isinstance(obj, firedrake.MeshGeometry):
        return obj, None
    raise TypeError(f"Cannot draw object of type {type(obj)}")


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


# UFC tetrahedron face data.
_UFC_TET_VERTS = np.array([[0, 0, 0], [1, 0, 0],
                           [0, 1, 0], [0, 0, 1]], dtype=np.float64)
_UFC_TET_FACET_VERTS = (
    (1, 2, 3),  # facet 0 opposite V0
    (0, 3, 2),  # facet 1 opposite V1  (V2/V3 swapped: outward normal)
    (0, 1, 3),  # facet 2 opposite V2
    (0, 2, 1),  # facet 3 opposite V3  (V1/V2 swapped: outward normal)
)

# Canonical P2-Lagrange points on the UFC reference tetrahedron — 4 corners
# followed by 6 edge midpoints in the order used by webgui's volume shader
# (matches NGSolve's ``makeP2Tets`` layout: V0, V1, V2, V3, mid(V0V3),
# mid(V1V3), mid(V2V3), mid(V0V1), mid(V0V2), mid(V1V2)).
_UFC_TET_P2_PTS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.5],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.5, 0.5, 0.0],
], dtype=np.float64)


def _facet_ref_pts_3d(local_facet, ref_pts_2d):
    """Map 2D reference triangle points to a face of the UFC tet."""
    f0, f1, f2 = _UFC_TET_FACET_VERTS[local_facet]
    V0 = _UFC_TET_VERTS[f0]
    V1 = _UFC_TET_VERTS[f1]
    V2 = _UFC_TET_VERTS[f2]
    s = ref_pts_2d[:, 0:1]
    t = ref_pts_2d[:, 1:2]
    return V0 + s * (V1 - V0) + t * (V2 - V0)


def _interp_for_vis(func, mesh, deg):
    """Return a sample-able function and its value rank.

    For Lagrange / DG-style spaces the original ``func`` is returned.
    Non-pointwise families (RT, Nedelec, BDM, ...) are interpolated
    into a *discontinuous* DG_deg / vector-DG_deg space so that any
    genuine cross-cell discontinuity in the field — only the
    normal/tangential trace of an H(div)/H(curl) field is continuous —
    is preserved by the renderer.
    """
    import firedrake
    elem = func.function_space().ufl_element()
    family = elem.family()
    shape = _value_shape(func)

    if family in _POINTEVAL_FAMILIES:
        return func, len(shape)

    if shape == ():
        V = firedrake.FunctionSpace(mesh, "DG", deg)
    else:
        vdim = int(np.prod(shape))
        V = firedrake.VectorFunctionSpace(mesh, "DG", deg, dim=vdim)

    target = firedrake.Function(V)
    try:
        target.interpolate(func)
    except Exception:
        target.project(func)
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
    phi = _tabulate(func, ref_pts)
    cnm = func.cell_node_map().values
    dofs = func.dat.data_ro[cnm]
    if phi.ndim == 2:                  # (ndof, npts)
        if dofs.ndim == 2:
            return np.einsum('cd,dn->cn', dofs, phi)
        return np.einsum('cdv,dn->cnv', dofs, phi)
    # vector basis (ndof, vdim, npts) — happens only when the caller
    # supplied a function in such a space directly.
    return np.einsum('cd,dvn->cnv', dofs, phi)


def _per_cell_coords(mesh, ref_pts):
    """Evaluate ``mesh.coordinates`` at reference points per cell."""
    coords = mesh.coordinates
    phi = _tabulate(coords, ref_pts)
    cnm = coords.cell_node_map().values
    dofs = coords.dat.data_ro[cnm]
    if phi.ndim == 2:
        return np.einsum('cdg,dn->cng', dofs, phi)
    return np.einsum('cd,dgn->cng', dofs, phi)


# ---------------------------------------------------------------------------
# Bernstein/Bézier helpers
# ---------------------------------------------------------------------------

def _bernstein_trig_lattice(og):
    """Bernstein-Lagrange points on the UFC reference triangle.

    Returns an ``(ndtrig, 2)`` array with rows ``(ix/og, iy/og)``
    iterated x-outer / y-inner. Matches the mode ordering in
    :func:`_bernstein_trig_inverse` so the inverse Bernstein matrix
    multiplication produces consistent control points that webgui's
    WebGL shader can consume.
    """
    pts = []
    for ix in range(og + 1):
        for iy in range(og + 1 - ix):
            pts.append((ix / og, iy / og))
    return np.asarray(pts, dtype=np.float64)


def _bernstein_seg_lattice(og):
    """1D Bernstein-Lagrange points on ``[0, 1]``."""
    return np.linspace(0.0, 1.0, og + 1, dtype=np.float64)


def _bernstein_trig_value(x, y, i, j, n):
    """Triangular Bernstein polynomial ``B^n_{ij}(x, y)``."""
    coef = math.factorial(n) / (
        math.factorial(i) * math.factorial(j) * math.factorial(n - i - j)
    )
    return coef * x**i * y**j * (1.0 - x - y)**(n - i - j)


def _bernstein_seg_value(x, j, n):
    """1D Bernstein polynomial ``B^n_j(x)``."""
    coef = math.factorial(n) / (math.factorial(j) * math.factorial(n - j))
    return coef * x**j * (1.0 - x)**(n - j)


@lru_cache(maxsize=8)
def _bernstein_trig_inverse(og):
    """Inverse Bernstein-triangle basis matrix at the canonical lattice."""
    ndtrig = (og + 1) * (og + 2) // 2
    Bvals = np.zeros((ndtrig, ndtrig), dtype=np.float64)
    ii = 0
    for ix in range(og + 1):
        for iy in range(og + 1 - ix):
            jj = 0
            for jx in range(og + 1):
                for jy in range(og + 1 - jx):
                    Bvals[ii, jj] = _bernstein_trig_value(
                        ix / og, iy / og, jx, jy, og)
                    jj += 1
            ii += 1
    return np.linalg.inv(Bvals)


@lru_cache(maxsize=8)
def _bernstein_seg_inverse(og):
    """Inverse 1D Bernstein basis matrix at the canonical lattice."""
    Bvals = np.zeros((og + 1, og + 1), dtype=np.float64)
    for i in range(og + 1):
        for j in range(og + 1):
            Bvals[i, j] = _bernstein_seg_value(i / og, j, og)
    return np.linalg.inv(Bvals)


def _sub_trig_patches(deg):
    """Affine maps from cubic-patch reference triangle to a source cell.

    Mirrors NGSolve's ``_make_trig`` tiling: for ``deg <= 3`` returns a
    single identity patch; for higher degrees the source triangle is
    split into ``n*n`` upward and ``(n-1)*(n-1)`` downward cubic
    patches, with ``n = (deg + 2) // 3``.

    Each entry is ``(origin, e1, e2)``: a cubic-patch point ``(s, t)``
    maps to ``origin + s*e1 + t*e2`` in the source cell's reference
    coordinates.
    """
    if deg <= _MAX_ORDER_2D:
        return [(np.array([0.0, 0.0]),
                 np.array([1.0, 0.0]),
                 np.array([0.0, 1.0]))]
    n = (deg + 2) // 3
    h = 1.0 / n
    patches = []
    for i in range(n):
        for j in range(n - i):
            patches.append((np.array([i * h, j * h]),
                            np.array([h, 0.0]),
                            np.array([0.0, h])))
    for i in range(n - 1):
        for j in range(n - i - 1):
            patches.append((np.array([(i + 1) * h, (j + 1) * h]),
                            np.array([-h, 0.0]),
                            np.array([0.0, -h])))
    return patches


def _scalar_field(vals):
    """Reduce a sampled function array to a single scalar per point.

    ``vals`` has shape ``(..., npts)`` for scalars or ``(..., npts, vdim)``
    for vectors. Vectors collapse via Euclidean norm so the colour bar
    always shows magnitude. Returns just the scalar array — ``funcdim``
    is always ``1`` from the renderer's point of view, since pmat[..., 3]
    carries the only function component we ship.
    """
    if vals.ndim == 2:
        return vals
    return np.linalg.norm(vals, axis=-1)


def _pad_to_3d(coord_pts):
    """Pad a per-cell coordinate array of shape ``(..., gdim)`` to ``(..., 3)``."""
    if coord_pts.shape[-1] == 3:
        return coord_pts.astype(np.float64, copy=False)
    pad = np.zeros(coord_pts.shape[:-1] + (3,), dtype=np.float64)
    pad[..., :coord_pts.shape[-1]] = coord_pts
    return pad


def _bernstein_patches(ref_pts_per_patch, mesh, func, ncells, og):
    """Pack ``(ncells * npatches, ndtrig, 4)`` of geometry+value samples.

    ``ref_pts_per_patch`` is an ``(npatches * ndtrig, 2)`` array of
    Bernstein-Lagrange points laid out patch-major in the source cell's
    reference triangle, where ``ndtrig = (og+1)(og+2)/2``.

    Returns a ``(ncells * npatches, ndtrig, 4)`` array ``[x, y, z, f]``
    plus the ``funcdim`` recorded for the original function (or ``0``
    if ``func is None``).
    """
    ndtrig = (og + 1) * (og + 2) // 2
    npatches = ref_pts_per_patch.shape[0] // ndtrig

    coord_pts = _per_cell_coords(mesh, ref_pts_per_patch)
    coord_pts_3d = _pad_to_3d(coord_pts)
    # (ncells, npatches, ndtrig, 3)
    coord_pts_3d = coord_pts_3d.reshape(ncells, npatches, ndtrig, 3)

    if func is not None:
        f_vis, _ = _interp_for_vis(func, mesh, max(og, 1))
        f_vals = _per_cell_eval(f_vis, ref_pts_per_patch)
        scalar_vals = _scalar_field(f_vals).reshape(ncells, npatches, ndtrig)
        funcdim = 1
    else:
        scalar_vals = np.zeros((ncells, npatches, ndtrig), dtype=np.float64)
        funcdim = 0

    pmat = np.empty((ncells, npatches, ndtrig, 4), dtype=np.float64)
    pmat[..., :3] = coord_pts_3d
    pmat[..., 3] = scalar_vals
    return pmat.reshape(ncells * npatches, ndtrig, 4), funcdim


def _encode_bezier_trigs(pmat_flat, og, encoding):
    """Convert per-patch Lagrange samples to encoded Bernstein control points.

    Input ``pmat_flat`` has shape ``(npatches_total, ndtrig, 4)``.
    Returns a list of ``ndtrig`` encoded ``(npatches_total, 4)`` arrays —
    NGSolve's ``Bezier_trig_points`` layout.
    """
    iB = _bernstein_trig_inverse(og)
    # ``BezierPnts[i, p, k] = sum_j iB[i, j] * pmat_flat[p, j, k]``
    bezier = np.einsum('ij,pjk->ipk', iB, pmat_flat).astype(np.float32)
    return [encodeData(bezier[i], np.float32, encoding)
            for i in range(bezier.shape[0])]


# ---------------------------------------------------------------------------
# Surface builders
# ---------------------------------------------------------------------------

def _build_bezier_2d(mesh, func, order, encoding):
    """Bernstein control points for a 2D triangle mesh.

    Returns ``(Bezier_trig_points, edges, funcdim, og)`` where
    ``Bezier_trig_points`` follows webgui's list-of-arrays layout,
    ``edges`` is a parallel list along the 1D mesh boundary, and
    ``og`` is the cubic-or-less Bernstein order used by the shader.
    """
    og = min(order, _MAX_ORDER_2D)
    ndtrig = (og + 1) * (og + 2) // 2

    patches = _sub_trig_patches(order)
    npatches = len(patches)
    cubic_lattice = _bernstein_trig_lattice(og)  # (ndtrig, 2)

    ref_pts_per_patch = np.empty((npatches * ndtrig, 2), dtype=np.float64)
    for k, (origin, e1, e2) in enumerate(patches):
        block = (origin
                 + cubic_lattice[:, 0:1] * e1
                 + cubic_lattice[:, 1:2] * e2)
        ref_pts_per_patch[k * ndtrig:(k + 1) * ndtrig] = block

    coords = mesh.coordinates
    ncells = coords.cell_node_map().values.shape[0]
    pmat_flat, funcdim = _bernstein_patches(
        ref_pts_per_patch, mesh, func, ncells, og)
    bezier_trig_points = _encode_bezier_trigs(pmat_flat, og, encoding)
    edges = _build_bezier_2d_edges(mesh, og, encoding)
    return bezier_trig_points, edges, funcdim, og


def _build_bezier_2d_edges(mesh, og, encoding):
    """1D Bernstein control points along the 2D mesh boundary."""
    ef = mesh.exterior_facets
    nsegs = int(ef.facet_cell.size)
    if nsegs == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return [encodeData(empty, np.float32, encoding) for _ in range(og + 1)]

    # Edge endpoints in the parent triangle's reference coords,
    # per local facet number.
    facet_endpoints = {
        0: (np.array([1.0, 0.0]), np.array([0.0, 1.0])),  # opp V0
        1: (np.array([0.0, 0.0]), np.array([0.0, 1.0])),  # opp V1
        2: (np.array([0.0, 0.0]), np.array([1.0, 0.0])),  # opp V2
    }
    seg_lattice = _bernstein_seg_lattice(og)             # (og+1,)

    facet_cells = ef.facet_cell[:, 0]
    local_facets = ef.local_facet_dat.data_ro

    ref_pts_all = np.empty((nsegs * (og + 1), 2), dtype=np.float64)
    for k in range(nsegs):
        a, b = facet_endpoints[int(local_facets[k])]
        ref_pts_all[k * (og + 1):(k + 1) * (og + 1)] = (
            a[None, :] + seg_lattice[:, None] * (b - a)[None, :])

    coords = mesh.coordinates
    phi = _tabulate(coords, ref_pts_all)         # (ndof_c, nsegs*(og+1))
    cdofs = coords.dat.data_ro[coords.cell_node_map().values]  # (ncells, ndof_c, gdim)
    # For each segment k, take its (og+1) points from the (nsegs*(og+1),) axis
    phi_per = phi.reshape(phi.shape[0], nsegs, og + 1)  # (ndof_c, nsegs, og+1)

    pts = np.einsum('cdg,dcn->cng', cdofs[facet_cells], phi_per).astype(np.float64)
    # ``pts`` shape: (nsegs, og+1, gdim)
    pts3d = _pad_to_3d(pts)

    pmat = np.zeros((nsegs, og + 1, 4), dtype=np.float64)
    pmat[..., :3] = pts3d
    iB = _bernstein_seg_inverse(og)
    edge_data = np.einsum('ij,sjk->isk', iB, pmat).astype(np.float32)
    return [encodeData(edge_data[i], np.float32, encoding)
            for i in range(og + 1)]


def _build_bezier_3d_surface(mesh, func, order, encoding):
    """Bernstein control points for the boundary of a 3D tet mesh."""
    ef = mesh.exterior_facets
    if ef.facet_cell.size == 0:
        og = min(order, _MAX_ORDER_2D)
        empty_trig = np.zeros((0, 4), dtype=np.float32)
        bez = [encodeData(empty_trig, np.float32, encoding)
               for _ in range((og + 1) * (og + 2) // 2)]
        edges = [encodeData(empty_trig, np.float32, encoding)
                 for _ in range(og + 1)]
        return bez, edges, 0, og

    og = min(order, _MAX_ORDER_2D)
    ndtrig = (og + 1) * (og + 2) // 2

    patches = _sub_trig_patches(order)
    npatches_per_face = len(patches)
    cubic_lattice = _bernstein_trig_lattice(og)

    # 2D ref-triangle points for each patch (patch-major).
    ref_pts_2d_per_patch = np.empty(
        (npatches_per_face * ndtrig, 2), dtype=np.float64)
    for k, (origin, e1, e2) in enumerate(patches):
        ref_pts_2d_per_patch[k * ndtrig:(k + 1) * ndtrig] = (
            origin + cubic_lattice[:, 0:1] * e1
                   + cubic_lattice[:, 1:2] * e2)

    facet_cells = ef.facet_cell[:, 0]
    local_facets = ef.local_facet_dat.data_ro
    nfacets = len(local_facets)

    coords = mesh.coordinates
    coord_dofs = coords.dat.data_ro[coords.cell_node_map().values]
    # Cache (ndof_c, npts) tabulations per local facet number.
    coord_phi = {
        f: _tabulate(coords, _facet_ref_pts_3d(f, ref_pts_2d_per_patch))
        for f in range(4)
    }

    f_vis = None
    func_phi = None
    func_dofs = None
    if func is not None:
        f_vis, _ = _interp_for_vis(func, mesh, max(og, 1))
        func_phi = {
            f: _tabulate(f_vis, _facet_ref_pts_3d(f, ref_pts_2d_per_patch))
            for f in range(4)
        }
        func_dofs = f_vis.dat.data_ro[f_vis.cell_node_map().values]

    npts_per_face = npatches_per_face * ndtrig
    pmat = np.zeros((nfacets * npatches_per_face, ndtrig, 4), dtype=np.float64)
    funcdim = 0
    for k in range(nfacets):
        cell = int(facet_cells[k])
        f = int(local_facets[k])
        phi_c = coord_phi[f]                      # (ndof_c, npts_per_face)
        c_dofs = coord_dofs[cell]                 # (ndof_c, 3)
        pts = np.einsum('dg,dn->ng', c_dofs, phi_c)  # (npts_per_face, 3)
        pmat[k * npatches_per_face:(k + 1) * npatches_per_face, :, :3] = (
            pts.reshape(npatches_per_face, ndtrig, 3))

        if func is not None:
            phi_f = func_phi[f]
            fd = func_dofs[cell]
            if fd.ndim == 1:                      # scalar dofs
                fvals = np.einsum('d,dn->n', fd, phi_f)
            elif fd.ndim == 2 and phi_f.ndim == 2:
                fvals_vec = np.einsum('dv,dn->nv', fd, phi_f)
                fvals = np.linalg.norm(fvals_vec, axis=-1)
            else:
                fvals_vec = np.einsum('d,dvn->nv', fd, phi_f)
                fvals = np.linalg.norm(fvals_vec, axis=-1)
            funcdim = 1
            pmat[k * npatches_per_face:(k + 1) * npatches_per_face, :, 3] = (
                fvals.reshape(npatches_per_face, ndtrig))

    bezier_trig_points = _encode_bezier_trigs(pmat, og, encoding)
    edges = _build_bezier_3d_edges(mesh, ef, og, encoding)
    return bezier_trig_points, edges, funcdim, og


def _build_bezier_3d_edges(mesh, ef, og, encoding):
    """Feature edges between boundary facets carrying different markers.

    Returns a list of ``og+1`` encoded ``(nseg, 4)`` arrays in NGSolve's
    Bernstein-edge layout.
    """
    facet_cells = ef.facet_cell[:, 0]
    local_facets = ef.local_facet_dat.data_ro
    nfacets = len(local_facets)
    if nfacets == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return [encodeData(empty, np.float32, encoding) for _ in range(og + 1)]

    coords = mesh.coordinates
    cnm = coords.cell_node_map().values
    parent_verts = cnm[facet_cells]

    # Coarse boundary-triangle in original vertex indices (3 verts/facet).
    bnd = np.empty((nfacets, 3), dtype=np.int32)
    for k in range(nfacets):
        bnd[k] = np.delete(parent_verts[k], int(local_facets[k]))

    edge_to_facets = {}
    for k in range(nfacets):
        tri = bnd[k]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (int(min(a, b)), int(max(a, b)))
            edge_to_facets.setdefault(key, []).append(k)

    # Markers per boundary facet.
    facet_markers = np.zeros(nfacets, dtype=np.int32)
    try:
        for m in ef.unique_markers:
            facet_markers[ef.subset(m).indices] = m
    except Exception:
        pass

    # In the local facet's 2D ref coords, the three vertices are at
    # (0,0), (1,0), (0,1); each (la, lb) pair maps to an affine segment.
    edge_endpoints_2d = {
        (0, 1): (np.array([0.0, 0.0]), np.array([1.0, 0.0])),
        (1, 2): (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        (0, 2): (np.array([0.0, 0.0]), np.array([0.0, 1.0])),
    }

    seg_lattice = _bernstein_seg_lattice(og)
    crease_segments = []                            # (ndof_c, npts) per segment
    crease_cells = []                               # parent cell index
    for edge, facets in edge_to_facets.items():
        markers = {int(facet_markers[k]) for k in facets}
        if len(markers) <= 1:
            continue
        k0 = facets[0]
        f = int(local_facets[k0])
        face_globals = _UFC_TET_FACET_VERTS[f]
        cell = int(facet_cells[k0])
        cell_verts = cnm[cell]
        local_for = {int(cell_verts[face_globals[i]]): i for i in range(3)}
        a, b = edge
        la = local_for[a]
        lb = local_for[b]
        key = (min(la, lb), max(la, lb))
        p_start_2d, p_end_2d = edge_endpoints_2d[key]
        if (la, lb) != key:
            p_start_2d, p_end_2d = p_end_2d, p_start_2d
        # Lattice points along this segment in the local facet's 2D ref.
        pts2d = (p_start_2d[None, :]
                 + seg_lattice[:, None] * (p_end_2d - p_start_2d)[None, :])
        pts3d_ref = _facet_ref_pts_3d(f, pts2d)
        crease_segments.append(pts3d_ref)
        crease_cells.append(cell)

    if not crease_segments:
        empty = np.zeros((0, 4), dtype=np.float32)
        return [encodeData(empty, np.float32, encoding) for _ in range(og + 1)]

    # Evaluate coordinates per segment via the parent cell's tabulation.
    seg_pts_3d = np.empty((len(crease_segments), og + 1, 3), dtype=np.float64)
    coord_dofs = coords.dat.data_ro[cnm]
    for k, (pts3d_ref, cell) in enumerate(zip(crease_segments, crease_cells)):
        phi_c = _tabulate(coords, pts3d_ref)
        seg_pts_3d[k] = np.einsum('dg,dn->ng', coord_dofs[cell], phi_c)

    pmat = np.zeros((seg_pts_3d.shape[0], og + 1, 4), dtype=np.float64)
    pmat[..., :3] = seg_pts_3d
    iB = _bernstein_seg_inverse(og)
    edge_data = np.einsum('ij,sjk->isk', iB, pmat).astype(np.float32)
    return [encodeData(edge_data[i], np.float32, encoding)
            for i in range(og + 1)]


# ---------------------------------------------------------------------------
# Volume builder (3D clipping shader input)
# ---------------------------------------------------------------------------

def _build_bezier_3d_volume(mesh, func, order3d, encoding):
    """``points3d`` for clipping a 3D tet mesh.

    Returns a list of ``np_per_tet`` encoded ``(ntets, 4)`` arrays where
    ``np_per_tet = 4`` for ``order3d == 1`` and ``10`` for ``order3d == 2``.
    """
    if order3d == 1:
        ref_pts = _UFC_TET_VERTS
    else:
        ref_pts = _UFC_TET_P2_PTS
    np_per_tet = ref_pts.shape[0]

    coord_pts = _per_cell_coords(mesh, ref_pts)    # (ncells, np_per_tet, 3)
    coord_pts_3d = _pad_to_3d(coord_pts)

    if func is not None:
        f_vis, _ = _interp_for_vis(func, mesh, max(order3d, 1))
        f_vals = _per_cell_eval(f_vis, ref_pts)
        scalar_vals = _scalar_field(f_vals)
    else:
        scalar_vals = np.zeros((coord_pts_3d.shape[0], np_per_tet),
                               dtype=np.float64)

    pmat = np.empty((coord_pts_3d.shape[0], np_per_tet, 4), dtype=np.float64)
    pmat[..., :3] = coord_pts_3d
    pmat[..., 3] = scalar_vals
    # Layout: list[np_per_tet] of (ntets, 4).
    out = [encodeData(pmat[:, i, :].astype(np.float32), np.float32, encoding)
           for i in range(np_per_tet)]
    return out


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def _func_minmax(func):
    if func is None:
        return 0.0, 0.0
    vals = func.dat.data_ro
    if vals.ndim > 1:
        vals = np.linalg.norm(vals, axis=1)
    return float(vals.min()), float(vals.max())


def _source_degree(func, mesh):
    """Pick the source-cell polynomial degree driving sub-triangulation."""
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
    return max(1, min(_MAX_SOURCE_DEGREE, max(deg, coord_deg)))


def _normalize_clipping(clipping):
    """Translate the ``clipping`` kwarg into webgui data-dict keys.

    Mirrors NGSolve's ``WebGLScene.GetData`` (lines 287-308 of
    ``netgen/webgui.py``): accepts ``True`` / ``False`` / ``None`` /
    ``dict``. Returns a dict of data-dict keys to merge in.
    """
    if clipping is None or clipping is False:
        return {}
    if clipping is True:
        return {"clipping": True}
    if not isinstance(clipping, dict):
        raise TypeError(f"Unsupported clipping spec: {type(clipping)!r}")

    out = {"clipping": True}
    spec = dict(clipping)
    if "vec" in spec:
        vec = spec.pop("vec")
        spec.setdefault("x", float(vec[0]))
        spec.setdefault("y", float(vec[1]))
        spec.setdefault("z", float(vec[2]))
    pnt = spec.pop("pnt", None)
    allowed = {"x", "y", "z", "dist", "function"}
    for name, val in spec.items():
        if name not in allowed:
            raise ValueError(
                f"Unsupported clipping key {name!r}; "
                f"allowed: {sorted(allowed | {'vec', 'pnt'})}")
        out[f"clipping_{name}"] = val
    if pnt is not None:
        out["_clipping_pnt"] = list(map(float, pnt))
    return out


class FiredrakeScene(BaseWebGuiScene):
    """A webgui scene for a Firedrake mesh or function.

    Parameters
    ----------
    obj : firedrake.MeshGeometry or firedrake.Function
        Object to visualise.
    mesh : firedrake.MeshGeometry, optional
        Override mesh (when ``obj`` is a function but you want to show
        it on a different mesh).
    order : int, optional
        Source polynomial degree driving sub-triangulation; surfaces use
        cubic Bernstein patches (capped at 3) and split source cells of
        higher degree into multiple patches. Defaults to the source
        element / coordinate degree (clamped to 10).
    colormap : str, list, ndarray, or None, optional
        Colour map used by the on-screen colour bar. Strings select a
        built-in (``"mana"`` (default), ``"cool_to_warm"``, ``"ngs"``);
        a stop list ``[(t, (r, g, b)), ...]`` or an ``(N, 3)`` RGB array
        is also accepted. Pass ``None`` to fall back to webgui's default
        rainbow.
    clipping : bool or dict, optional
        Enable webgui's clipping plane. ``True`` enables with defaults;
        a dict may set any of ``x``, ``y``, ``z``, ``dist``, ``function``
        (replaces those individual keys), plus the conveniences ``vec``
        (3-vector → x/y/z) and ``pnt`` (overrides ``mesh_center``).
        Only meaningful for 3D meshes.
    draw_vol : bool, optional
        For 3D meshes, ship volume samples (``points3d``) so the
        clipping shader has something to slice. Default ``True`` when a
        function is provided. Ignored in 2D.
    """

    def __init__(self, obj, mesh=None, order=None,
                 colormap=_DEFAULT_COLORMAP,
                 clipping=None,
                 draw_vol=True,
                 **kwargs):
        # Back-compat: accept the old ``subdivision`` kwarg as an alias.
        if order is None and "subdivision" in kwargs:
            order = kwargs.pop("subdivision")
        self.obj = obj
        self._mesh_override = mesh
        self.order = order
        self.colormap = colormap
        self.clipping = clipping
        self.draw_vol = draw_vol
        self.kwargs = kwargs
        self.encoding = "b64"

    def _resolve(self):
        mesh, func = _get_mesh_and_func(self.obj)
        if self._mesh_override is not None:
            mesh = self._mesh_override
        order = self.order or _source_degree(func, mesh)
        order = max(1, min(_MAX_SOURCE_DEGREE, int(order)))
        return mesh, func, order

    def GetData(self, set_minmax=True):
        mesh, func, order = self._resolve()
        tdim = mesh.topological_dimension

        if tdim == 2:
            bez_trigs, edges, funcdim, og2 = _build_bezier_2d(
                mesh, func, order, self.encoding)
            order3d = 0
            points3d = None
        elif tdim == 3:
            bez_trigs, edges, funcdim, og2 = _build_bezier_3d_surface(
                mesh, func, order, self.encoding)
            order3d = min(order, _MAX_ORDER_3D)
            if func is not None and self.draw_vol:
                points3d = _build_bezier_3d_volume(
                    mesh, func, order3d, self.encoding)
            else:
                points3d = None
        else:
            raise ValueError(f"Unsupported topological dimension {tdim}")

        # Bounding box from the geometry samples (using the first three
        # entries of each Bezier_trig_points array would round-trip
        # base64; cheaper to derive it directly from coordinates here).
        cdata = mesh.coordinates.dat.data_ro
        if cdata.ndim == 1:
            cdata = cdata.reshape(-1, 1)
        gdim = cdata.shape[1]
        vmin = np.zeros(3, dtype=np.float64)
        vmax = np.zeros(3, dtype=np.float64)
        vmin[:gdim] = cdata.min(axis=0)
        vmax[:gdim] = cdata.max(axis=0)
        center = ((vmin + vmax) / 2.0).tolist()
        radius = float(np.linalg.norm(vmax - vmin) / 2.0)

        funcmin, funcmax = _func_minmax(func)

        d = {
            "ngsolve_version": "firedrake_webgui",
            "mesh_dim": int(tdim),
            "mesh_center": center,
            "mesh_radius": radius,
            "order2d": int(og2),
            "order3d": int(order3d) if tdim == 3 else 0,
            "draw_surf": True,
            "draw_vol": bool(points3d is not None),
            "show_wireframe": True,
            "show_mesh": True,
            "Bezier_trig_points": bez_trigs,
            "edges": edges,
            "funcdim": int(funcdim),
            "is_complex": False,
            "autoscale": bool(set_minmax),
            "funcmin": float(funcmin),
            "funcmax": float(funcmax),
        }
        if points3d is not None:
            d["points3d"] = points3d

        colors = _sample_colormap(self.colormap)
        if colors is not None:
            d["colors"] = colors

        clip_extras = _normalize_clipping(self.clipping)
        pnt = clip_extras.pop("_clipping_pnt", None)
        if pnt is not None:
            d["mesh_center"] = pnt
        d.update(clip_extras)

        d.update({k: v for k, v in self.kwargs.items()
                  if k in ("settings", "autoscale", "min", "max",
                           "draw_vol", "draw_surf",
                           "show_wireframe", "show_mesh")})
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


def Draw(obj, mesh=None, order=None,
         colormap=_DEFAULT_COLORMAP,
         clipping=None,
         draw_vol=True,
         **kwargs):
    """Draw a Firedrake Mesh or Function in Jupyter."""
    width = kwargs.pop("width", None)
    height = kwargs.pop("height", None)
    if order is None and "subdivision" in kwargs:
        order = kwargs.pop("subdivision")
    scene = FiredrakeScene(obj, mesh=mesh, order=order,
                           colormap=colormap, clipping=clipping,
                           draw_vol=draw_vol, **kwargs)
    scene.Draw(width=width, height=height)
    return scene
