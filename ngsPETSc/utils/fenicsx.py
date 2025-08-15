"""
This module contains all the functions related to wrapping NGSolve meshes to FEniCSx
We adopt the same docstring conventiona as the FEniCSx project, since this part of
the package will only be used in combination with FEniCSx.
"""

import typing
import basix.ufl
import dolfinx
import numpy as np
import numpy.typing as npt
import ufl
from packaging.version import Version
from mpi4py import MPI as _MPI
from ngsPETSc.utils.utils import find_permutation
from ngsPETSc import MeshMapping

# Map from Netgen cell type (integer tuple) to GMSH cell type
_ngs_to_cells = {(2, 3): 2, (2, 4): 3, (3, 4): 4}


def _dim_to_element_wrapper(ngmesh: typing.Any) -> dict[int, typing.Any]:
    """Convenience wrapper to extract elements from a NetGen mesh based on topological dimension"""
    return {
        1: ngmesh.Elements1D,
        2: ngmesh.Elements2D,
        3: ngmesh.Elements3D,
    }


class GeometricModel:
    """
    This class is used to wrap a Netgen geometric model to a DOLFINx mesh.
    Args:
            geo: The Netgen model
            comm: The MPI communicator to use for mesh creation
    """

    def __init__(self, geo, comm: _MPI.Comm, comm_rank: int = 0):
        self.geo = geo
        self.comm = comm
        self.comm_rank = comm_rank

    def model_to_mesh(
        self,
        hmax: float,
        gdim: int = 2,
        partitioner: typing.Callable[
            [_MPI.Comm, int, int, dolfinx.cpp.graph.AdjacencyList_int32],
            dolfinx.cpp.graph.AdjacencyList_int32,
        ] = dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet),
        transform: typing.Any = None,
        routine: typing.Any = None,
        meshing_options: dict[str, typing.Any] | None = None,
    ) -> tuple[
        dolfinx.mesh.Mesh,
        tuple[dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags],
        dict[tuple[int, str], tuple[int, ...]],
    ]:
        """Given a NetGen model, take all physical entities of the highest
        topological dimension and create the corresponding linear DOLFINx mesh.

        Args:
            hmax: The maximum diameter of the elements in the triangulation
            model: The NetGen model
            gdim: Geometrical dimension of the mesh
            partitioner: Function that computes the parallel
                distribution of cells across MPI ranks
            transform: PETSc DMPLEX Transformation to be applied to the mesh
            routine: Function to be applied to the mesh after generation
                takes as plan the mesh and the NetGen model and returns the
                same objects after the routine has been applied.

        Returns:
            A DOLFINx mesh for the given NetGen model. It also extracts cell tags,
            facet tags and a mapping from the NetGen label to the corresponding integer marker(s).
        """
        meshing_options = {} if meshing_options is None else meshing_options

        # To be parallel safe, we generate on all processes
        # NOTE: This might change in the future.

        # First we generate a mesh
        ngmesh = self.geo.GenerateMesh(maxh=hmax, **meshing_options)
        self.ngmesh = ngmesh
        # Apply any ngs routine post meshing
        if routine is not None:
            ngmesh, self.geo = routine(ngmesh, self.geo)
        if transform is not None:
            meshMap = MeshMapping(ngmesh)
            transform.setDM(meshMap.plex)
            transform.setUp()
            newplex = transform.apply(meshMap.plex)
            meshMap = MeshMapping(newplex)
            ngmesh = meshMap.ngmesh


        assert ngmesh.dim in (2, 3), "Only 2D and 3D meshes are supported."
        regions = self.extract_regions()
        mesh, ct, ft = self.extract_linear_mesh(gdim=gdim, partitioner=partitioner)
        return mesh, (ct, ft), regions

    def extract_regions(self):
        """Extract regions from the Netgen mesh."""
        ngmesh = self.ngmesh
        # Create lookup from cells/facets materials to integer tags
        regions: dict[tuple[int, str], list[int]] = {}
        # Append facet material
        for dim in (ngmesh.dim, ngmesh.dim - 1):
            for name in ngmesh.GetRegionNames(dim=dim):
                regions[(dim, name)] = []
            for i, name in enumerate(ngmesh.GetRegionNames(dim=dim)):
                regions[(dim, name)].append(i + 1)

        for key, value in regions.items():
            regions[key] = tuple(value)
        return regions

    def extract_linear_mesh(
        self,
        gdim: int,
        partitioner=dolfinx.mesh.create_cell_partitioner(
            dolfinx.mesh.GhostMode.shared_facet
        ),
    ) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags]:
        """
        Extract a DOLFINx mesh (and correpsonding cell and facet tags) from the Netgen mesh.
        """
        V, T = None, None
        ngmesh = self.ngmesh
        elements_as_numpy = _dim_to_element_wrapper(ngmesh)[ngmesh.dim]().NumPy()
        T = elements_as_numpy["nodes"]

        number_of_vertices = elements_as_numpy["np"]
        sorted_index = np.argsort(number_of_vertices)
        offset = number_of_vertices[sorted_index[1:]]-number_of_vertices[sorted_index[:-1]]
        cell_boundaries = np.flatnonzero(offset)+1
        if len(cell_boundaries) == 0:
            mixed_mesh = False
            split = -1
        else:
            mixed_mesh = True
            assert len(cell_boundaries) == 1, "Only two different cell types are expected in a 2D grid."
            split = cell_boundaries[0]
        
        if self.comm.rank == self.comm_rank:
            # Applying any PETSc Transform
            # We extract topology and geometry
            V = ngmesh.Coordinates()

            if mixed_mesh:
                num_points_per_cell = [number_of_vertices[sorted_index[0]], number_of_vertices[sorted_index[cell_boundaries[0]]]]
                cell_offsets = np.array([0, num_points_per_cell[1]+1, elements_as_numpy.shape[0]])
                _T = []
                for i in range(len(num_points_per_cell)):
                    if Version(np.__version__) >= Version("2.2"):
                        _T.append(np.trim_zeros(T[cell_offsets[i]:cell_offsets[i+1]], "b", axis=1).astype(np.int64) - 1)
                    else:
                        _T.append(
                            np.array(
                                [list(np.trim_zeros(a, "b")) for a in list(T[cell_offsets[i]:cell_offsets[i+1]])],
                                dtype=np.int64,
                            ) - 1
                            
                        )

                T = _T


                breakpoint()
                pass
            else:
                if Version(np.__version__) >= Version("2.2"):
                    T = np.trim_zeros(T, "b", axis=1).astype(np.int64) - 1
                else:


                    T = (
                        np.array(
                            [list(np.trim_zeros(a, "b")) for a in list(T)],
                            dtype=np.int64,
                        )
                        - 1
                    )
        else:
            if mixed_mesh:
                pass
            else:
                V = np.zeros((0, 3), dtype=np.float64)
                T = np.zeros((0, number_of_vertices[0]), dtype=np.int64)
        
        ufl_domain = dolfinx.io.gmshio.ufl_mesh(
            _ngs_to_cells[(gdim, T.shape[1])], gdim, dolfinx.default_real_type
        )

        cell_str = ufl_domain.ufl_coordinate_element().cell_type.name
        T = T[:, dolfinx.cpp.io.perm_vtk(dolfinx.mesh.to_type(cell_str), T.shape[1])]

        mesh = dolfinx.mesh.create_mesh(
            self.comm, cells=T, x=V[:,:gdim], e=ufl_domain, partitioner=partitioner
        )
        self._mesh = mesh

        ct = extract_element_tags(self.comm_rank, ngmesh, mesh, dim=ngmesh.dim)
        ct.name = "Cell tags"
        ft = extract_element_tags(self.comm_rank, ngmesh, mesh, dim=ngmesh.dim - 1)
        ft.name = "Facet tags"
        return mesh, ct, ft

    def curveField(
        self, order: int, permutation_tol: float = 1e-8, location_tol: float = 1e-10
    ):
        """
        This method returns a curved mesh as a Firedrake function.

        :arg order: the order of the curved mesh.
        :arg permutation_tol: tolerance used to construct the permutation of the reference element.
        :arg location_tol: tolerance used to locate the cell a point belongs to.
        """
        if self._mesh.geometry.cmap.degree == order:
            return self._mesh

        dim_to_element_getter = _dim_to_element_wrapper(self.ngmesh)
        ng_element = dim_to_element_getter[self.ngmesh.dim]
        ng_dimension = len(ng_element())  # Number of cells in NGS grid (on any rank)
        geom_dim = self.ngmesh.dim

        el = basix.ufl.element(
            "Lagrange", self._mesh.basix_cell(), order, shape=(geom_dim,)
        )

        rsp = el.basix_element.x  # NOTE: FIX empty points in basix nanobind wrapper
        reference_space_points = []
        for lin in rsp:
            if len(lin) != 0:
                reference_space_points.append(np.vstack(lin))
        reference_space_points = np.vstack(reference_space_points)

        # Curve the mesh on rank 0 only
        if self.comm.rank == self.comm_rank:
            # Construct numpy arrays for physical domain data
            physical_space_points = np.zeros(
                (ng_dimension, reference_space_points.shape[0], geom_dim)
            )
            curved_space_points = np.zeros(
                (ng_dimension, reference_space_points.shape[0], geom_dim)
            )
            self.ngmesh.CalcElementMapping(
                reference_space_points, physical_space_points
            )
            self.ngmesh.Curve(order)
            self.ngmesh.CalcElementMapping(reference_space_points, curved_space_points)
            curved = ng_element().NumPy()["curved"]
            # Broadcast a boolean array identifying curved cells
            curved = self.comm.bcast(curved, root=self.comm_rank)
            physical_space_points = physical_space_points[curved]
            curved_space_points = curved_space_points[curved]
        else:
            curved = self.comm.bcast(None, root=self.comm_rank)
            # Construct numpy arrays as buffers to receive physical domain data
            ncurved = np.sum(curved)
            physical_space_points = np.zeros(
                (ncurved, reference_space_points.shape[0], geom_dim)
            )
            curved_space_points = np.zeros(
                (ncurved, reference_space_points.shape[0], geom_dim)
            )

        # Broadcast curved cell point data
        self.comm.Bcast(physical_space_points, root=self.comm_rank)
        self.comm.Bcast(curved_space_points, root=self.comm_rank)

        # Get coordinates of higher order space on linarized geometry
        X_space = dolfinx.fem.functionspace(self._mesh, el)
        x = X_space.tabulate_dof_coordinates()  # Shape (num_nodes, 3)
        cell_map = self._mesh.topology.index_map(self._mesh.topology.dim)
        num_cells_owned = cell_map.size_local + cell_map.num_ghosts
        cell_node_map = X_space.dofmap.list[:num_cells_owned]
        new_coordinates = x[cell_node_map]

        # Collision detection of barycenter of cell
        psp_shape = physical_space_points.shape
        padded_physical_space_points = np.zeros(
            (psp_shape[0], psp_shape[1], 3), dtype=physical_space_points.dtype
        )
        padded_physical_space_points[:, :, :geom_dim] = physical_space_points

        # Barycenters of curved cells (exists on all processes)
        barycentres = np.average(padded_physical_space_points, axis=1)

        # Create bounding box for function evaluation
        bb_tree = dolfinx.geometry.bb_tree(
            self._mesh,
            self._mesh.topology.dim,
            np.arange(num_cells_owned, dtype=np.int32),
            padding=location_tol,
        )

        # Check against standard table value
        cell_candidates = dolfinx.geometry.compute_collisions_points(
            bb_tree, barycentres
        )
        owned = dolfinx.geometry.compute_colliding_cells(
            self._mesh, cell_candidates, barycentres
        )
        owned_pos = np.flatnonzero(owned.offsets[1:] - owned.offsets[:-1])
        owned_cells = owned.array
        assert len(owned_cells) == len(owned_pos)

        # NOTE: There should be an algorithm for this
        # Find the correct coordinate permutation for each cell
        if len(owned_pos) > 0:
            owned_psp = padded_physical_space_points[owned_pos]
            permutation = find_permutation(
                owned_psp,
                new_coordinates[owned_cells]
                .reshape(owned_psp.shape)
                .astype(self._mesh.geometry.x.dtype, copy=False),
                tol=permutation_tol,
            )
        else:
            permutation = np.zeros(
                (0, padded_physical_space_points.shape[1]), dtype=np.int64
            )

        # Apply the permutation to each cell in turn
        if len(owned_cells) > 0:
            for ii, p in enumerate(curved_space_points[owned_pos]):
                curved_space_points[owned_pos[ii]] = p[permutation[ii]]
            # Assign the curved coordinates to the dat
            x[cell_node_map[owned_cells].flatten(), :geom_dim] = curved_space_points[
                owned_pos
            ].reshape(-1, geom_dim)

        # Use topology from original mesh
        topology = self._mesh.topology
        # Use geometry from function_space
        c_el = dolfinx.fem.coordinate_element(el.basix_element)  #  pylint: disable=E1120
        geom_imap = X_space.dofmap.index_map
        local_node_indices = np.arange(
            geom_imap.size_local + geom_imap.num_ghosts, dtype=np.int32
        )
        igi = geom_imap.local_to_global(local_node_indices)
        geometry = dolfinx.mesh.create_geometry(
            geom_imap, cell_node_map, c_el._cpp_object, x[:, :geom_dim].copy(), igi
        )

        # Create DOLFINx mesh
        if x.dtype == np.float64:
            cpp_mesh = dolfinx.cpp.mesh.Mesh_float64(
                self._mesh.comm, topology._cpp_object, geometry._cpp_object
            )
        else:
            raise RuntimeError(f"Unsupported dtype for mesh {x.dtype}")
        # Wrap as Python object
        return dolfinx.mesh.Mesh(cpp_mesh, domain=ufl.Mesh(el))

    def refineMarkedElements(
        self,
        dim: int,
        elements: npt.NDArray[np.int32],
        netgen_flags: dict | None = None,
    ):
        """Refine mesh based on marked elements."""
        netgen_flags = netgen_flags or {}
        refine_faces = netgen_flags.get("refine_faces", False)
        gdim = self._mesh.geometry.dim
        if gdim not in (2, 3):
            raise RuntimeError("Refinement of 2D and 3D meshes is supported only.")
        # Gather all the element indices for refinement on rank 0.
        # Map elements to incidient cells.
        self._mesh.topology.create_connectivity(dim, self._mesh.topology.dim)
        local_cells = dolfinx.mesh.compute_incident_entities(
            self._mesh.topology, elements, dim, self._mesh.topology.dim
        )
        igi = self._mesh.topology.original_cell_index[local_cells]
        gathered_igi = self._mesh.comm.gather(igi, root=self.comm_rank)

        if self._mesh.comm.rank == self.comm_rank:
            ng_elements = _dim_to_element_wrapper(self.ngmesh)[
                self._mesh.topology.dim
            ]()
            ng_dimension = len(ng_elements)
            marker = np.zeros(ng_dimension, dtype=np.int8)
            marker[np.hstack(gathered_igi)] = 1
            for refine, el in zip(marker, ng_elements, strict=True):
                if refine:
                    el.refine = True
                else:
                    el.refine = False
            if not refine_faces and dim == 3:
                _dim_to_element_wrapper(self.ngmesh)[2]().Numpy()["refine"] = 0
            self.ngmesh.Refine(adaptive=True)
        self.ngmesh.Curve(1)  # Reset mesh to be linear

        self._mesh.comm.Barrier()
        mesh, ct, ft = self.extract_linear_mesh(gdim=gdim)
        return mesh, (ct, ft)


def extract_element_tags(
    comm_rank: int, ngmesh, dolfinx_mesh: dolfinx.mesh.Mesh, dim: int
) -> dolfinx.mesh.MeshTags:
    """
    Extract element tags from a Netgen mesh (on a given MPI rank) and distribute them onto
    the corresponding DOLFINx mesh.

    Args:
        comm_rank: The MPI rank to extract the element from.
        ngmesh: The Netgen mesh object.
        dolfinx_mesh: The DOLFINx mesh to which the facet tags will be distributed to.
        dim: The topological dimension of the entities to extract.
    """
    tdim = dolfinx_mesh.topology.dim
    assert ngmesh.dim == tdim, f"Mismatch: ({ngmesh.dim=}!={tdim=})"
    assert dolfinx_mesh.geometry.cmap.degree == 1, "Can only extract element tags from linear grids"
    comm = dolfinx_mesh.comm
    sub_entities = basix.cell.subentity_types(dolfinx_mesh.basix_cell())[dim]
    assert len(np.unique(sub_entities)) == 1, "Only one subentity type is supported"
    entity_type = dolfinx.mesh.to_type(sub_entities[0].name)
    num_vertices_per_cell = dolfinx.cpp.mesh.cell_num_vertices(entity_type)
    if comm.rank == comm_rank:
        ng_entities = _dim_to_element_wrapper(ngmesh)[dim]()
        element_indices = ng_entities.NumPy()["nodes"].astype(np.int64)
        if Version(np.__version__) >= Version("2.2"):
            entitites = np.trim_zeros(element_indices, "b", axis=1).astype(np.int64) - 1
        else:
            entitites = (
                np.array(
                    [list(np.trim_zeros(a, "b")) for a in list(element_indices)],
                    dtype=np.int64,
                )
                - 1
            )
        if dim == dolfinx_mesh.topology.dim:
            entity_markers = ng_entities.NumPy()["index"].astype(np.int32)
        else:
            # Can't use the vectorized version, due to a bug in ngsolve:
            # https://forum.ngsolve.org/t/extract-facet-markers-from-netgen-mesh/3256
            entity_markers = np.array(
                [entity.index for entity in ng_entities], dtype=np.int32
            )
    else:
        # NOTE: Mixed meshes on non-simplex geometries requires changes
        entitites = np.zeros((0, num_vertices_per_cell), dtype=np.int64)
        entity_markers = np.zeros((0,), dtype=np.int32)

    local_entities, local_values = dolfinx.io.gmshio.distribute_entity_data(
        dolfinx_mesh, dim, entitites, entity_markers
    )
    dolfinx_mesh.topology.create_connectivity(dim, 0)
    adj = dolfinx.graph.adjacencylist(local_entities)
    meshtag = dolfinx.mesh.meshtags_from_entities(
        dolfinx_mesh,
        dim,
        adj,
        local_values,
    )
    return meshtag
