from netgen.occ import *
import numpy as np

from firedrake.mg import HierarchyBase
from firedrake import *
from fractions import Fraction
from firedrake.mg.utils import set_level, get_level
from collections import defaultdict

__all__ = ["AdaptiveMeshHierarchy"]


class AdaptiveMeshHierarchy(HierarchyBase):
    def __init__(self, mesh, refinements_per_level=1, nested=True):
        self.meshes = tuple(mesh)
        self._meshes = tuple(mesh)
        self.submesh_hierarchies = []
        self.coarse_to_fine_cells = None
        self.fine_to_coarse_cells = None
        self.refinements_per_level = refinements_per_level
        self.nested = nested
        set_level(mesh[0], self, 0)
        self.split_cache = {}

    def add_mesh(self, mesh):
        if mesh.topological_dimension() <= 2:
            max_children = 4
        else:
            max_children = 16
        self._meshes += tuple(mesh)
        self.meshes += tuple(mesh)
        coarse_mesh = self.meshes[-2]
        level = len(self.meshes)
        set_level(self.meshes[-1], self, level - 1)
        self._shared_data_cache = defaultdict(dict)

        if len(self.meshes) <= 2:
            self.coarse_to_fine_cells = {}
            self.fine_to_coarse_cells = {}
            self.fine_to_coarse_cells[Fraction(0, 1)] = None

        # extract parent child relationships from netgen meshes,
        (c2f, f2c) = get_c2f_f2c_fd(mesh, coarse_mesh)

        self.coarse_to_fine_cells[Fraction(len(self.meshes) - 2, 1)] = c2f
        self.fine_to_coarse_cells[Fraction(len(self.meshes) - 1, 1)] = np.array(f2c)

        # split both the fine and coarse meshes into the submeshes
        (coarse_splits, fine_splits, num_children) = split_to_submesh(
            mesh, coarse_mesh, c2f, f2c
        )
        for i in range(1, max_children + 1):
            coarse_mesh.mark_entities(coarse_splits[i], i)
            mesh.mark_entities(fine_splits[i], int(f"10{i}"))

        coarse_mesh = RelabeledMesh(
            coarse_mesh,
            [coarse_splits[i] for i in range(1, max_children + 1)],
            list(range(1, max_children + 1)),
            name="Relabeled_coarse",
        )
        c_subm = {
            j: Submesh(coarse_mesh, coarse_mesh.topology_dm.getDimension(), j)
            for j in range(1, max_children + 1)
            if any(num_children == j)
        }
        set_level(coarse_mesh, self, level - 2)

        mesh = RelabeledMesh(
            mesh,
            [fine_splits[i] for i in range(1, max_children + 1)],
            list(range(1, max_children + 1)),
        )
        f_subm = {
            int(str(j)[-2:]): Submesh(mesh, mesh.topology_dm.getDimension(), j)
            for j in [int("10" + str(i)) for i in range(1, max_children + 1)]
            if any(num_children == int(str(j)[-2:]))
        }
        set_level(mesh, self, level - 1)

        # update c2f and f2c for submeshes by mapping numberings on full mesh to numberings on coarse mesh
        n = [
            len([el for el in c2f if len(el) == j]) for j in range(1, max_children + 1)
        ]
        c2f_adjusted = {
            j: np.zeros((n, j))
            for n, j in zip(n, list(range(1, max_children + 1)))
            if n != 0
        }
        f2c_adjusted = {
            j: np.zeros((n * j, 1))
            for n, j in zip(n, list(range(1, max_children + 1)))
            if n != 0
        }

        coarse_full_to_sub_map = {
            i: full_to_sub(coarse_mesh, c_subm[i], i) for i in c_subm
        }
        fine_full_to_sub_map = {
            j: full_to_sub(mesh, f_subm[j], int("10" + str(j))) for j in f_subm
        }

        for i in range(len(c2f)):
            n = len(c2f[i])
            if 1 <= n <= max_children:
                c2f_adjusted[n][coarse_full_to_sub_map[n](i)] = fine_full_to_sub_map[n](
                    np.array(c2f[i])
                )

        for j in range(len(f2c)):
            n = int(num_children[f2c[j]].item())
            if 1 <= n <= max_children:
                f2c_adjusted[n][fine_full_to_sub_map[n](j), 0] = coarse_full_to_sub_map[
                    n
                ](f2c[j].item())

        c2f_subm = {
            i: {Fraction(0, 1): c2f_adjusted[i].astype(int)} for i in c2f_adjusted
        }
        f2c_subm = {i: {Fraction(1, 1): f2c_adjusted[i]} for i in f2c_adjusted}

        hierarchy_dict = {
            i: HierarchyBase(
                [c_subm[i], f_subm[i]], c2f_subm[i], f2c_subm[i], nested=True
            )
            for i in c_subm
        }
        self.submesh_hierarchies.append(hierarchy_dict)

    def refine(self, refinements):
        # Input a list corresponding to which elements get refined
        ngmesh = self.meshes[-1].netgen_mesh
        for l, el in enumerate(ngmesh.Elements2D()):
            el.refine = 0
            if refinements[l] == 1:
                el.refine = 1

        ngmesh.Refine(adaptive=True)
        mesh = Mesh(ngmesh)
        self.add_mesh(mesh)

    def adapt(self, eta, theta):
        # Implement Dorfler marking, returns new mesh
        mesh = self.meshes[-1]
        W = FunctionSpace(mesh, "DG", 0)
        markers = Function(W)

        with eta.dat.vec_ro as eta_:
            eta_max = eta_.max()[1]

        should_refine = conditional(gt(eta, theta * eta_max), 1, 0)
        markers.interpolate(should_refine)

        refined_mesh = mesh.refine_marked_elements(markers)
        self.add_mesh(refined_mesh)
        return refined_mesh

    def split_function(self, u, child=True):
        # Split function across submeshes
        V = u.function_space()
        full_mesh = V.mesh()
        _, level = get_level(full_mesh)

        ind = 1 if child else 0
        hierarchy_dict = self.submesh_hierarchies[int(level) - ind]
        u_corr_space = Function(
            V.reconstruct(
                hierarchy_dict[[*hierarchy_dict][0]].meshes[ind].submesh_parent
            ),
            val=u.dat,
        )
        key = (u, child)
        try:
            split_functions = self.split_cache[key]
        except KeyError:
            split_functions = self.split_cache.setdefault(key, {})

        for i in hierarchy_dict:
            try:
                f = split_functions[i].zero()
            except KeyError:
                V_split = V.reconstruct(mesh=hierarchy_dict[i].meshes[ind])
                assert (
                    V_split.mesh().submesh_parent
                    == u_corr_space.function_space().mesh()
                )
                f = split_functions.setdefault(i, Function(V_split, name=str(i)))

            f.assign(u_corr_space)
        return split_functions

    def use_weight(self, V, child):
        # counts of DoFs across submeshes, to fix restriction before recombinatoin
        w = Function(V).assign(1)
        splits = self.split_function(w, child)

        self.recombine(splits, w, child)
        with w.dat.vec as wvec:
            wvec.reciprocal()
        return w

    def recombine(self, split_funcs, f, child=True):
        # Recombines functions on submeshes to full mesh
        V = f.function_space()
        f.zero()
        parent_mesh = (
            split_funcs[[*split_funcs][0]].function_space().mesh().submesh_parent
        )
        V_label = V.reconstruct(mesh=parent_mesh)
        if isinstance(f, Function):
            f_label = Function(V_label, val=f.dat)
        elif isinstance(f, Cofunction):
            f_label = Cofunction(V_label, val=f.dat)

        for split_label, val in split_funcs.items():
            assert val.function_space().mesh().submesh_parent == parent_mesh
            if child:
                split_label = int("10" + str(split_label))
            if isinstance(f_label, Function):
                f_label.assign(val, allow_missing_dofs=True)
            else:
                curr = Function(f_label.function_space()).assign(
                    val, allow_missing_dofs=True
                )
                f_label.assign(f_label + curr)  # partition of unity for restriction
        return f


def get_c2f_f2c_fd(mesh, coarse_mesh):
    # Construct coarse -> fine and fine -> coarse relations by mapping netgen elements to fd
    V = FunctionSpace(mesh, "DG", 0)
    V2 = FunctionSpace(coarse_mesh, "DG", 0)
    ngmesh = mesh.netgen_mesh
    num_parents = coarse_mesh.num_cells()

    if mesh.topology_dm.getDimension() == 2:
        parents = ngmesh.parentsurfaceelements.NumPy()
        elements = ngmesh.Elements2D()
    if mesh.topology_dm.getDimension() == 3:
        parents = ngmesh.parentelements.NumPy()
        elements = ngmesh.Elements3D()
    fine_mapping = lambda x: mesh._cell_numbering.getOffset(x)
    coarse_mapping = lambda x: coarse_mesh._cell_numbering.getOffset(x)

    c2f = [[] for _ in range(num_parents)]
    f2c = [[] for _ in range(mesh.num_cells())]

    if parents.shape[0] == 0:
        raise Exception("Added mesh has not refined any cells from previous mesh")
    for l, _ in enumerate(elements):
        if parents[l][0] == -1 or l < num_parents:
            f2c[fine_mapping(l)].append(coarse_mapping(l))
            c2f[coarse_mapping(l)].append(fine_mapping(l))

        elif parents[l][0] < num_parents:
            f2c[fine_mapping(l)].append(coarse_mapping(parents[l][0]))
            c2f[coarse_mapping(parents[l][0])].append(fine_mapping(l))

        else:
            a = parents[parents[l][0]][0]
            while a >= num_parents:
                a = parents[a][0]

            f2c[fine_mapping(l)].append(coarse_mapping(a))
            c2f[coarse_mapping(a)].append(fine_mapping(l))

    return c2f, np.array(f2c).astype(int)


def split_to_submesh(mesh, coarse_mesh, c2f, f2c):
    # Splits mesh into element numberings to generate submeshes
    if mesh.topological_dimension() <= 2:
        max_children = 4
    else:
        max_children = 16
    V = FunctionSpace(mesh, "DG", 0)
    V2 = FunctionSpace(coarse_mesh, "DG", 0)
    coarse_splits = {
        i: Function(V2, name=f"{i}_elements") for i in range(1, max_children + 1)
    }
    fine_splits = {
        i: Function(V, name=f"{i}_elements") for i in range(1, max_children + 1)
    }
    num_children = np.zeros((len(c2f)))

    for i in range(len(c2f)):
        n = len(c2f[i])
        if 1 <= n <= max_children:
            coarse_splits[n].dat.data[i] = 1
            num_children[i] = n

    for i in range(1, max_children + 1):
        fine_splits[i].dat.data[num_children[f2c.squeeze()] == i] = 1

    return coarse_splits, fine_splits, num_children


def full_to_sub(mesh, submesh, label):
    # returns the submesh element number associated with the full mesh numbering
    V1 = FunctionSpace(mesh, "DG", 0)
    V2 = FunctionSpace(submesh, "DG", 0)
    u1 = Function(V1)
    u2 = Function(V2)
    u2.dat.data[:] = np.arange(len(u2.dat.data))
    u1.interpolate(u2, subset=mesh.cell_subset(label))

    return lambda x: u1.dat.data[x].astype(int)
