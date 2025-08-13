'''
This module contains all the functions related to wrapping NGSolve meshes to FEniCSx
We adopt the same docstring conventiona as the FEniCSx project, since this part of
the package will only be used in combination with FEniCSx.
'''
import typing
import dolfinx
import numpy as np
from packaging.version import Version
from mpi4py import MPI as _MPI

from ngsPETSc import MeshMapping

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
        dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.none),
        transform: typing.Any = None, routine: typing.Any = None
        ) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dict[str, tuple[int, ...]]]:
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
            A DOLFINx mesh for the given NetGen model.
        """
        # To be parallel safe, we generate on all processes
        # NOTE: This might change in the future.

        # First we generate a mesh
        ngmesh = self.geo.GenerateMesh(maxh=hmax)
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
            regions: dict[str, list[int]] = {name: []
                                             for name in ngmesh.GetRegionNames(dim=ngmesh.dim-1)}
            for i, name in enumerate(ngmesh.GetRegionNames(ngmesh.dim-1), 1):
                regions[name].append(i)

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
            regions = self.comm.bcast(regions, root=0)
        else:
            # NOTE: Mixed meshes on non-simplex geometries requires changes
            facets = np.zeros((0, ngmesh.dim), dtype=np.int64)
            facet_values = np.zeros((0,), dtype=np.int32)
            regions = self.comm.bcast(None, root=0)

        local_entities, local_values = dolfinx.io.gmshio.distribute_entity_data(
            mesh, mesh.topology.dim - 1, facets, facet_values
        )
        mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
        adj = dolfinx.graph.adjacencylist(local_entities)
        ft = dolfinx.mesh.meshtags_from_entities(
            mesh,
            mesh.topology.dim - 1,
            adj,
            local_values.astype(np.int32, copy=False),
        )
        ft.name = "Facet tags"

        for key, value in regions.items():
            regions[key] = tuple(value)
        return mesh, ft, regions
