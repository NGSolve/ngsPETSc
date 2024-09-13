'''
This module contains all the functions related to wrapping NGSolve meshes to Firedrake
We adopt the same docstring conventiona as the Firedrake project, since this part of
the package will only be used in combination with Firedrake.
'''
try:
    import firedrake as fd
    from firedrake.__future__ import interpolate
except ImportError:
    fd = None

import numpy as np
from petsc4py import PETSc

import netgen
import netgen.meshing as ngm
from netgen.meshing import MeshingParameters
try:
    import ngsolve as ngs
except ImportError:
    class ngs:
        "dummy class"
        class comp:
            "dummy class"
            Mesh = type(None)

from ngsPETSc import MeshMapping

def flagsUtils(flags, option, default):
    '''
    utility fuction used to parse Netgen flag options
    '''
    try:
        return flags[option]
    except KeyError:
        return default

def refineMarkedElements(self, mark):
    '''
    This method is used to refine a mesh based on a marking function
    which is a Firedrake DG0 function.

    :arg mark: the marking function which is a Firedrake DG0 function.

    '''
    els = {2: self.netgen_mesh.Elements2D, 3: self.netgen_mesh.Elements3D}
    dim = self.geometric_dimension()
    if dim in [2,3]:
        with mark.dat.vec as marked:
            marked0 = marked
            getIdx = self._cell_numbering.getOffset
            if self.sfBCInv is not None:
                getIdx = lambda x: x #pylint: disable=C3001
                _, marked0 = self.topology_dm.distributeField(self.sfBCInv,
                                                              self._cell_numbering,
                                                              marked)
            if self.comm.Get_rank() == 0:
                mark = marked0.getArray()
                max_refs = np.max(mark)
                for _ in range(int(max_refs)):
                    for i, el in enumerate(els[dim]()):
                        if mark[getIdx(i)] > 0:
                            el.refine = True
                        else:
                            el.refine = False
                    self.netgen_mesh.Refine(adaptive=True)
                    mark = mark-np.ones(mark.shape)
                return fd.Mesh(self.netgen_mesh)
            return fd.Mesh(netgen.libngpy._meshing.Mesh(dim))
    else:
        raise NotImplementedError("No implementation for dimension other than 2 and 3.")

def curveField(self, order, tol=1e-8):
    '''
    This method returns a curved mesh as a Firedrake function.

    :arg order: the order of the curved mesh

    '''
    #Checking if the mesh is a surface mesh or two dimensional mesh
    surf = len(self.netgen_mesh.Elements3D()) == 0
    #Constructing mesh as a function
    low_order_element = self.coordinates.function_space().ufl_element().sub_elements[0]
    element = low_order_element.reconstruct(degree=order)
    space = fd.VectorFunctionSpace(self, fd.BrokenElement(element))
    newFunctionCoordinates = fd.assemble(interpolate(self.coordinates, space))
    #Computing reference points using fiat
    fiat_element = newFunctionCoordinates.function_space().finat_element.fiat_equivalent
    entity_ids = fiat_element.entity_dofs()
    nodes = fiat_element.dual_basis()
    refPts = []
    for dim in entity_ids:
        for entity in entity_ids[dim]:
            for dof in entity_ids[dim][entity]:
                # Assert singleton point for each node.
                pt, = nodes[dof].get_point_dict().keys()
                refPts.append(pt)
    V = newFunctionCoordinates.dat.data
    refPts = np.array(refPts)
    els = {True: self.netgen_mesh.Elements2D, False: self.netgen_mesh.Elements3D}
    #Mapping to the physical domain
    if self.comm.rank == 0:
        physPts = np.ndarray((len(els[surf]()),
                                refPts.shape[0], self.geometric_dimension()))
        self.netgen_mesh.CalcElementMapping(refPts, physPts)
        #Cruving the mesh
        self.netgen_mesh.Curve(order)
        curvedPhysPts = np.ndarray((len(els[surf]()),
                                    refPts.shape[0], self.geometric_dimension()))
        self.netgen_mesh.CalcElementMapping(refPts, curvedPhysPts)
        curved = els[surf]().NumPy()["curved"]
    else:
        physPts = np.ndarray((len(els[surf]()),
                                refPts.shape[0], self.geometric_dimension()))
        curvedPhysPts = np.ndarray((len(els[surf]()),
                                    refPts.shape[0], self.geometric_dimension()))
        curved = np.array((len(els[surf]()),1))
    physPts = self.comm.bcast(physPts, root=0)
    curvedPhysPts = self.comm.bcast(curvedPhysPts, root=0)
    curved = self.comm.bcast(curved, root=0)
    cellMap = newFunctionCoordinates.cell_node_map()
    for i in range(physPts.shape[0]):
        #Inefficent code but runs only on curved elements
        if curved[i]:
            pts = physPts[i][0:refPts.shape[0]]
            bary = sum([np.array(pts[i]) for i in range(len(pts))])/len(pts)
            Idx = self.locate_cell(bary)
            isInMesh = (0<=Idx<len(cellMap.values)) if Idx is not None else False
            #Check if element is shared across processes
            shared = self.comm.gather(isInMesh, root=0)
            shared = self.comm.bcast(shared, root=0)
            #Bend if not shared
            if np.sum(shared) == 1:
                if isInMesh:
                    p = [np.argmin(np.sum((pts - pt)**2, axis=1))
                            for pt in V[cellMap.values[Idx]][0:refPts.shape[0]]]
                    curvedPhysPts[i] = curvedPhysPts[i][p]
                    res = np.linalg.norm(pts[p]-V[cellMap.values[Idx]][0:refPts.shape[0]])
                    if res > tol:
                        fd.logging.warning("[{}] Not able to curve Firedrake element {} \
                            ({}) -- residual: {}".format(self.comm.rank, Idx,i, res))
                    else:
                        for j, datIdx in enumerate(cellMap.values[Idx][0:refPts.shape[0]]):
                            for dim in range(self.geometric_dimension()):
                                coo = curvedPhysPts[i][j][dim]
                                newFunctionCoordinates.sub(dim).dat.data[datIdx] = coo
            else:
                if isInMesh:
                    p = [np.argmin(np.sum((pts - pt)**2, axis=1))
                            for pt in V[cellMap.values[Idx]][0:refPts.shape[0]]]
                    curvedPhysPts[i] = curvedPhysPts[i][p]
                    res = np.linalg.norm(pts[p]-V[cellMap.values[Idx]][0:refPts.shape[0]])
                else:
                    res = np.inf
                res = self.comm.gather(res, root=0)
                res = self.comm.bcast(res, root=0)
                rank = np.argmin(res)
                if self.comm.rank == rank:
                    if res[rank] > tol:
                        fd.logging.warning("[{}, {}] Not able to curve Firedrake element {} \
                            ({}) -- residual: {}".format(self.comm.rank, shared, Idx,i, res))
                    else:
                        for j, datIdx in enumerate(cellMap.values[Idx][0:refPts.shape[0]]):
                            for dim in range(self.geometric_dimension()):
                                coo = curvedPhysPts[i][j][dim]
                                newFunctionCoordinates.sub(dim).dat.data[datIdx] = coo

    return newFunctionCoordinates

def splitToQuads(plex, dim, comm):
    '''
    This method splits a Netgen mesh to quads, using a PETSc transform.
    TODO: Improve support quad meshing.
        @pef  Get netgen to make a quad-dominant mesh, and then only split the triangles.
              Current implementation will make for poor-quality meshes.
    '''
    if dim == 2:
        transform = PETSc.DMPlexTransform().create(comm=comm)
        transform.setType(PETSc.DMPlexTransformType.REFINETOBOX)
        transform.setDM(plex)
        transform.setUp()
    else:
        raise RuntimeError("Splitting to quads is only possible for 2D meshes.")
    newplex = transform.apply(plex)
    return newplex

splitTypes = {"Alfeld": lambda x: x.SplitAlfeld(),
              "Powell-Sabin": lambda x: x.SplitPowellSabin()}

class FiredrakeMesh:
    '''
    This class creates a Firedrake mesh from Netgen/NGSolve meshes.

    :arg mesh: the mesh object, it can be either a Netgen/NGSolve mesh or a PETSc DMPlex
    :param netgen_flags: The dictionary of flags to be passed to ngsPETSc.
    :arg comm: the MPI communicator.
    '''
    def __init__(self, mesh, netgen_flags, user_comm=fd.COMM_WORLD):
        self.comm = user_comm
        #Parsing netgen flags
        if not isinstance(netgen_flags, dict):
            netgen_flags = {}
        split2tets = flagsUtils(netgen_flags, "split_to_tets", False)
        split = flagsUtils(netgen_flags, "split", False)
        quad = flagsUtils(netgen_flags, "quad", False)
        optMoves = flagsUtils(netgen_flags, "optimisation_moves", False)
        #Checking the mesh format
        if isinstance(mesh,(ngs.comp.Mesh,ngm.Mesh)):
            if split2tets:
                mesh = mesh.Split2Tets()
            if split:
                #Split mesh this includes Alfeld and Powell-Sabin
                splitTypes[split](mesh)
            if optMoves:
                #Optimises the mesh, for example smoothing
                if mesh.dim == 2:
                    mesh.OptimizeMesh2d(MeshingParameters(optimize2d=optMoves))
                elif mesh.dim == 3:
                    mesh.OptimizeVolumeMesh(MeshingParameters(optimize3d=optMoves))
                else:
                    raise ValueError("Only 2D and 3D meshes can be optimised.")
            #We create the plex from the netgen mesh
            self.meshMap = MeshMapping(mesh, comm=self.comm)
            #We apply the DMPLEX transform
            if quad:
                newplex = splitToQuads(self.meshMap.petscPlex, mesh.dim, comm=self.comm)
                self.meshMap = MeshMapping(newplex)
        else:
            raise ValueError("Mesh format not recognised.")

    def createFromTopology(self, topology, name, comm):
        '''
        Internal method to construct a mesh from a mesh topology, copied from Firedrake.

        :arg topology: the mesh topology

        :arg name: the mesh name

        '''
        cell = topology.ufl_cell()
        geometric_dim = topology.topology_dm.getCoordinateDim()
        cell = cell.reconstruct(geometric_dimension=geometric_dim)
        element = fd.VectorElement("Lagrange", cell, 1)
        # Create mesh object
        self.firedrakeMesh = fd.MeshGeometry.__new__(fd.MeshGeometry, element, comm)
        self.firedrakeMesh._init_topology(topology)
        self.firedrakeMesh.name = name
        # Adding Netgen mesh and inverse sfBC as attributes
        self.firedrakeMesh.netgen_mesh = self.meshMap.ngMesh
        if self.firedrakeMesh.sfBC is not None:
            self.firedrakeMesh.sfBCInv = self.firedrakeMesh.sfBC.createInverse()
        else:
            self.firedrakeMesh.sfBCInv = None
        #Generating ngs to Firedrake cell index map
        #Adding refine_marked_elements and curve_field methods
        setattr(fd.MeshGeometry, "refine_marked_elements", refineMarkedElements)
        setattr(fd.MeshGeometry, "curve_field", curveField)
        #Adding labels for boundary regions and regions
        self.firedrakeMesh.labels = {}
        for dim in range(1, geometric_dim+1):
            for i, label in enumerate(self.firedrakeMesh.netgen_mesh.GetRegionNames(dim=dim)):
                if (dim, label) not in self.firedrakeMesh.labels:
                    self.firedrakeMesh.labels[(dim, label)] = ()
                self.firedrakeMesh.labels[(dim, label)] \
                    = (*self.firedrakeMesh.labels[(dim, label)],i+1)
                if  dim == geometric_dim \
                    and len(self.firedrakeMesh.netgen_mesh.GetRegionNames(dim=dim)) > 1:
                    offset = len(self.firedrakeMesh.netgen_mesh.GetRegionNames(dim=dim-1))
                    if (dim-1, label) not in self.firedrakeMesh.labels:
                        self.firedrakeMesh.labels[(dim-1, label)] = ()
                    self.firedrakeMesh.labels[(dim-1, label)] \
                        = (*self.firedrakeMesh.labels[(dim-1, label)],i+1+offset)
