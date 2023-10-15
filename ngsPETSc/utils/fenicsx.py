'''
This module contains all the functions related to wrapping NGSolve meshes to FEniCSx
We adopt the same docstring conventiona as the FEniCSx project, since this part of
the package will only be used in combination with FEniCSx.
'''
try:
    import dolfinx
    import basix
    import ufl
except ImportError:
    dolfinx = None
    basix = None
    ufl = None

import typing 
import numpy as np

import netgen as ng
import netgen.meshing as ngm

from mpi4py import MPI as _MPI

from ngsPETSc import MeshMapping

# Map from Netgen cell type (integer tuple) to DOLFINx cell type
_ngs_to_cells = {(2,3): ("triangle", 1),
                 (2,4): ("quadrilateral", 1),
                 (3,4): ("tetrahedron", 1)}

def ufl_mesh(ngs_cell: int, gdim: int) -> ufl.Mesh:
    """Create a UFL mesh from a Netgen cell identifier and the geometric dimension.
    Args:
        ngs_cell: The netgen cell identifier
        gdim: The geometric dimension of the mesh

    Returns:
        A ufl Mesh using Lagrange elements (equispaced) of the
        corresponding DOLFINx cell
    """
    shape, degree = _ngs_to_cells[ngs_cell]
    cell = ufl.Cell(shape, geometric_dimension=gdim)

    element = basix.ufl.element(
        basix.ElementFamily.P, cell.cellname(), degree, basix.LagrangeVariant.equispaced, shape=(gdim, ),
        gdim=gdim)
    return ufl.Mesh(element)

class GeometricModel:
    """
    This class is used to wrap a Netgen geometric model to a DOLFINx mesh.
    Args:
            geo: The Netgen model
            comm: The MPI communicator to use for mesh creation
    """
    def __init__(self,geo, comm: _MPI.Comm):
            self.geo = geo
            self.comm = comm
    
    def model_to_mesh(self, hmax: float, gdim: int = 2,
                        partitioner: typing.Callable[
            [_MPI.Comm, int, int, dolfinx.cpp.graph.AdjacencyList_int32], dolfinx.cpp.graph.AdjacencyList_int32] =
            dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.none), transform: typing.Any = None,
            routine: typing.Any = None) -> typing.Tuple[dolfinx.mesh.Mesh, dolfinx.cpp.mesh.MeshTags_int32,
            dolfinx.cpp.mesh.MeshTags_int32]:
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
        # First we generate a mesh
        ngmesh = self.geo.GenerateMesh(maxh=hmax)
        # Apply any ngs routine post meshing
        if routine is not None:
            ngmesh, geo = routine(ngmesh, geo)
        # Applying any PETSc Transform
        if transform is not None:
            meshMap = MeshMapping(ngmesh) 
            transform.setDM(meshMap.plex)
            transform.setUp()
            newplex = transform.apply(meshMap.plex)
            meshMap = MeshMapping(newplex)
            ngmesh = meshMap.ngmesh
        # We extract topology and geometry
        if ngmesh.dim == 2:
            V = ngmesh.Coordinates()
            T = ngmesh.Elements2D().NumPy()["nodes"]
            T = np.array([list(np.trim_zeros(a, 'b')) for a in list(T)])-1
        elif ngmesh.dim == 3:
            V = ngmesh.Coordinates()
            T = ngmesh.Elements3D().NumPy()["nodes"]
            T = np.array([list(np.trim_zeros(a, 'b')) for a in list(T)])-1
        ufl_domain = ufl_mesh((gdim,T.shape[1]),gdim)
        cell_perm = dolfinx.cpp.io.perm_gmsh(dolfinx.cpp.mesh.to_type(str(ufl_domain.ufl_cell())), T.shape[1])
        T = T[:, cell_perm]
        mesh = dolfinx.mesh.create_mesh(self.comm, T, V, ufl_domain, partitioner)
        return mesh

