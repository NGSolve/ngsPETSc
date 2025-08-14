'''
This module contains all the functions related to wrapping NGSolve meshes to FEniCSx
We adopt the same docstring conventiona as the FEniCSx project, since this part of
the package will only be used in combination with FEniCSx.
'''
import typing
import basix.ufl
import dolfinx
import numpy as np
from packaging.version import Version
from mpi4py import MPI as _MPI
from ngsPETSc.utils.utils import find_permutation
from ngsPETSc import MeshMapping
import ufl
# Map from Netgen cell type (integer tuple) to GMSH cell type
_ngs_to_cells = {(2,3): 2, (2,4):3, (3,4): 4}


class GeometricModel:
    """
    This class is used to wrap a Netgen geometric model to a DOLFINx mesh.
    Args:
            geo: The Netgen model
            comm: The MPI communicator to use for mesh creation
    """
    def __init__(self,geo, comm: _MPI.Comm, comm_rank:int = 0):
        self.geo = geo
        self.comm = comm
        self.comm_rank = comm_rank

    def model_to_mesh(self, hmax: float, gdim: int = 2,
        partitioner: typing.Callable[
        [_MPI.Comm, int, int, dolfinx.cpp.graph.AdjacencyList_int32],
        dolfinx.cpp.graph.AdjacencyList_int32] =
        dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet),
        transform: typing.Any = None, routine: typing.Any = None
        ) -> tuple[dolfinx.mesh.Mesh, tuple[dolfinx.mesh.MeshTags,dolfinx.mesh.MeshTags],
                   dict[tuple[int, str], tuple[int, ...]]]:
        """Given a NetGen model, take all physical entities of the highest
        topological dimension and create the corresponding DOLFINx mesh.

        This function only works in serial, at the moment.

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
        # To be parallel safe, we generate on all processes
        # NOTE: This might change in the future.

        # First we generate a mesh
        ngmesh = self.geo.GenerateMesh(maxh=hmax)
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
        _dim_to_element_wrapper = {
            1: ngmesh.Elements1D,
            2: ngmesh.Elements2D,
            3: ngmesh.Elements3D}

        V, T = None, None
        if self.comm.rank == self.comm_rank:

            # Applying any PETSc Transform
            # We extract topology and geometry
            V = ngmesh.Coordinates()
            T = _dim_to_element_wrapper[ngmesh.dim]().NumPy()["nodes"]
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
            # NOTE: For mixed meshes, this must change
            V = np.zeros((0, ngmesh.dim), dtype=np.float64)
            T = np.zeros((0, ngmesh.dim+1), dtype=np.int64)

        # NOTE: Here we should curve meshes
        ufl_domain = dolfinx.io.gmshio.ufl_mesh(
            _ngs_to_cells[(gdim,T.shape[1])], gdim, dolfinx.default_real_type)
        cell_perm = dolfinx.cpp.io.perm_gmsh(dolfinx.cpp.mesh.to_type(str(ufl_domain.ufl_cell())),
                                             T.shape[1])
        T = np.ascontiguousarray(T[:, cell_perm])
        mesh = dolfinx.mesh.create_mesh(self.comm, cells=T, x=V, e=ufl_domain,
                                        partitioner=partitioner)

        if self.comm.rank == self.comm_rank:
            cell_values = _dim_to_element_wrapper[ngmesh.dim]().NumPy()["index"].astype(np.int32)
        else:
            cell_values = np.zeros((0,), dtype=np.int32)

        local_entities, local_values = dolfinx.io.gmshio.distribute_entity_data(
            mesh, mesh.topology.dim, T, cell_values)
        adj_cells = dolfinx.graph.adjacencylist(local_entities)
        ct = dolfinx.mesh.meshtags_from_entities(
            mesh,
            mesh.topology.dim,
            adj_cells,
            local_values,
        )
        ct.name = "Cell tags"

        # Create lookup from cells/facets materials to integer tags
        regions: dict[tuple[int, str], list[int]] = {}
        # Append facet material
        for dim in (ngmesh.dim, ngmesh.dim - 1):
            for name in ngmesh.GetRegionNames(dim=dim):
                regions[(dim, name)] = []
            for i, name in enumerate(ngmesh.GetRegionNames(dim=dim)):
                regions[(dim, name)].append(i + 1)


        if self.comm.rank == self.comm_rank:

            ng_facets = _dim_to_element_wrapper[ngmesh.dim-1]()
            facet_indices = ng_facets.NumPy()["nodes"].astype(np.int64)
            if Version(np.__version__) >= Version("2.2"):
                facets = np.trim_zeros(facet_indices, "b", axis=1).astype(np.int64) - 1
            else:
                facets = (
                    np.array(
                        [list(np.trim_zeros(a, "b")) for a in list(facet_indices)],
                        dtype=np.int64,
                    )
                    - 1
                )
            # Can't use the vectorized version, due to a bug in ngsolve:
            # https://forum.ngsolve.org/t/extract-facet-markers-from-netgen-mesh/3256
            facet_values = np.array([facet.index for facet in ng_facets], dtype=np.int32)
        else:
            # NOTE: Mixed meshes on non-simplex geometries requires changes
            facets = np.zeros((0, ngmesh.dim), dtype=np.int64)
            facet_values = np.zeros((0,), dtype=np.int32)

        local_entities, local_values = dolfinx.io.gmshio.distribute_entity_data(
            mesh, mesh.topology.dim - 1, facets, facet_values
        )
        mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
        adj = dolfinx.graph.adjacencylist(local_entities)
        ft = dolfinx.mesh.meshtags_from_entities(
            mesh,
            mesh.topology.dim - 1,
            adj,
            local_values,
        )
        ft.name = "Facet tags"

        for key, value in regions.items():
            regions[key] = tuple(value)
        self._mesh = mesh
        self._tags = (ct, ft)
        self._regions = regions

        return mesh, (ct, ft), regions

    def curveField(self, order:int, permutation_tol:float=1e-8, location_tol:float=1e-1):
        '''
        This method returns a curved mesh as a Firedrake function.

        :arg order: the order of the curved mesh.
        :arg permutation_tol: tolerance used to construct the permutation of the reference element.
        :arg location_tol: tolerance used to locate the cell a point belongs to.
        '''
        # Check if the mesh is a surface mesh or two dimensional mesh

        _dim_to_element_wrapper = {
            1: self.ngmesh.Elements1D,
            2: self.ngmesh.Elements2D,
            3: self.ngmesh.Elements3D}

        ng_element = _dim_to_element_wrapper[self.ngmesh.dim]
        ng_dimension = len(ng_element()) # Number of cells in NGS grid (on any rank)
        geom_dim = self.ngmesh.dim


        el = basix.ufl.element("Lagrange", self._mesh.basix_cell(), order, shape=(geom_dim, ))

        rsp = el.basix_element.x # NOTE: FIX empty points in basix nanobind wrapper
        reference_space_points = []
        for lin in rsp:
            if len(lin) != 0:
                reference_space_points.append(np.vstack(lin))
        reference_space_points = np.vstack(reference_space_points)

        # Curve the mesh on rank 0 only
        if self.comm.rank == 0:
            # Construct numpy arrays for physical domain data
            physical_space_points = np.zeros(
                (ng_dimension, reference_space_points.shape[0], geom_dim)
            )
            curved_space_points = np.zeros(
                (ng_dimension, reference_space_points.shape[0], geom_dim)
            )
            self.ngmesh.CalcElementMapping(reference_space_points, physical_space_points)
            self.ngmesh.Curve(order)
            self.ngmesh.CalcElementMapping(reference_space_points, curved_space_points)
            curved = ng_element().NumPy()["curved"]
            # Broadcast a boolean array identifying curved cells
            curved = self.comm.bcast(curved, root=0)
            physical_space_points = physical_space_points[curved]
            curved_space_points = curved_space_points[curved]
        else:
            curved = self.comm.bcast(None, root=0)
            # Construct numpy arrays as buffers to receive physical domain data
            ncurved = np.sum(curved)
            physical_space_points = np.zeros(
                (ncurved, reference_space_points.shape[0], geom_dim)
            )
            curved_space_points = np.zeros(
                (ncurved, reference_space_points.shape[0], geom_dim)
            )

        # Broadcast curved cell point data
        self.comm.Bcast(physical_space_points, root=0)
        self.comm.Bcast(curved_space_points, root=0)

        # Get coordinates of higher order space on linarized geometry
        X_space = dolfinx.fem.functionspace(self._mesh, el)
        x = X_space.tabulate_dof_coordinates() # Shape (num_nodes, 3)
        cell_map = self._mesh.topology.index_map(self._mesh.topology.dim)
        num_cells_owned = cell_map.size_local + cell_map.num_ghosts
        cell_node_map = X_space.dofmap.list[:num_cells_owned]
        new_coordinates = x[cell_node_map]

        # Collision detection of barycenter of cell
        psp_shape = physical_space_points.shape
        padded_physical_space_points = np.zeros((psp_shape[0],
                                                 psp_shape[1], 3),
                                                 dtype=physical_space_points.dtype)
        padded_physical_space_points[:, :, :geom_dim] = physical_space_points

        # Barycenters of curved cells (exists on all processes)
        barycentres = np.average(padded_physical_space_points, axis=1)

        # Create bounding box for function evaluation
        bb_tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.topology.dim,
                                           np.arange(num_cells_owned, dtype=np.int32))

        # Check against standard table value
        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, barycentres)
        owned = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, barycentres)
        owned_pos = np.flatnonzero(owned.offsets[1:]-owned.offsets[:-1])
        owned_cells = owned.array
        assert len(owned_cells) == len(owned_pos)

        # Find the correct coordinate permutation for each cell
        if len(owned_pos) > 0:
            owned_psp = padded_physical_space_points[owned_pos]
            permutation = find_permutation(
                owned_psp,
                new_coordinates[owned_cells].reshape(
                    owned_psp.shape
                ).astype(self._mesh.geometry.x.dtype, copy=False),
                tol=permutation_tol
            )
        else:
            permutation = np.zeros((0, padded_physical_space_points.shape[1]), dtype=np.int64)
        # Apply the permutation to each cell in turn
        if len(owned_cells) > 0:
            for ii, p in enumerate(curved_space_points[owned_pos]):
                curved_space_points[owned_pos[ii]] = p[permutation[ii]]
            # Assign the curved coordinates to the dat
            x[cell_node_map[owned_cells].flatten(), :geom_dim] = curved_space_points[owned_pos].reshape(-1, geom_dim)

        # Sync ghosted coordinates across all processes
        coord_func = dolfinx.fem.Function(X_space)
        num_dofs_local = X_space.dofmap.index_map.size_local * X_space.dofmap.index_map_bs
        coord_func.x.array[:] = x[:num_dofs_local, :geom_dim].flatten()
        coord_func.x.scatter_forward()

        # Use topology from original mesh
        topology = self._mesh.topology
        # Use geometry from function_space
        c_el = dolfinx.fem.coordinate_element(el.basix_element)
        geom_imap = X_space.dofmap.index_map
        local_node_indices = np.arange(geom_imap.size_local + geom_imap.num_ghosts, dtype=np.int32)
        igi = geom_imap.local_to_global(local_node_indices)
        geometry = dolfinx.mesh.create_geometry(geom_imap, cell_node_map, c_el._cpp_object,
                                                coord_func.x.array.reshape(-1, geom_dim), igi)

        # Create DOLFINx mesh
        if x.dtype == np.float64:
            cpp_mesh = dolfinx.cpp.mesh.Mesh_float64(self._mesh.comm, topology._cpp_object,
                                                     geometry._cpp_object)
        else:
            raise RuntimeError(f"Unsupported dtype for mesh {x.dtype}")
        # Wrap as Python object
        return dolfinx.mesh.Mesh(cpp_mesh, domain=ufl.Mesh(el))