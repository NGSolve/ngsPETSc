'''
This module contains all the functions related to wrapping NGSolve meshes to Firedrake
We adopt the same docstring conventiona as the Firedrake project, since this part of
the package will only be used in combination with Firedrake.
'''
try:
    import firedrake as fd
    import firedrake.mg as mg
    import ufl
except ImportError:
    fd = None
    ufl = None
    mg = None

import warnings
import numpy as np
from petsc4py import PETSc

import netgen
import netgen.meshing as ngm
try:
    import ngsolve as ngs
except ImportError:
    class ngs:
        "dummy class"
        class comp:
            "dummy class"
            Mesh = type(None)

from ngsPETSc import MeshMapping

def refineMarkedElements(self, mark):
    '''
    This method is used to refine a mesh based on a marking function
    which is a Firedrake DG0 function.

    :arg mark: the marking function which is a Firedrake DG0 function.

    '''
    return NetgenHierarchy.refineMarkedElements(self, mark)[0]
def curveField(self, order):
    '''
    This method returns a curved mesh as a Firedrake funciton.

    :arg order: the order of the curved mesh

    '''
    low_order_element = self.coordinates.function_space().ufl_element().sub_elements()[0]
    element = low_order_element.reconstruct(degree=order)
    space = fd.VectorFunctionSpace(self, ufl.BrokenElement(element))
    newFunctionCoordinates = fd.interpolate(self.coordinates, space)

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
    getIdx = self._cell_numbering.getOffset
    refPts = np.array(refPts)
    rnd = lambda x: round(x, 8)
    if self.geometric_dimension() == 2:
        #Mapping to the physical domain
        physPts = np.ndarray((len(self.netgen_mesh.Elements2D()),
                             refPts.shape[0], self.geometric_dimension()))
        self.netgen_mesh.CalcElementMapping(refPts, physPts)
        #Cruving the mesh
        self.netgen_mesh.Curve(order)
        curvedPhysPts = np.ndarray((len(self.netgen_mesh.Elements2D()),
                                   refPts.shape[0], self.geometric_dimension()))
        self.netgen_mesh.CalcElementMapping(refPts, curvedPhysPts)
        cellMap = newFunctionCoordinates.cell_node_map()
        for i, el in enumerate(self.netgen_mesh.Elements2D()):
            if el.curved:
                pts = [tuple(map(rnd, pts))
                       for pts in physPts[i][0:refPts.shape[0]]]
                dofMap = {k: v for v, k in enumerate(pts)}
                p = [dofMap[tuple(map(rnd, pts))]
                     for pts in V[cellMap.values[getIdx(i)]][0:refPts.shape[0]]]
                curvedPhysPts[i] = curvedPhysPts[i][p]
                for j, datIdx in enumerate(cellMap.values[getIdx(i)][0:refPts.shape[0]]):
                    newFunctionCoordinates.sub(0).dat.data[datIdx] = curvedPhysPts[i][j][0]
                    newFunctionCoordinates.sub(1).dat.data[datIdx] = curvedPhysPts[i][j][1]

    if self.geometric_dimension() == 3:
        #Mapping to the physical domain
        physPts = np.ndarray((len(self.netgen_mesh.Elements3D()),
                             refPts.shape[0], self.geometric_dimension()))
        self.netgen_mesh.CalcElementMapping(refPts, physPts)
        #Cruving the mesh
        self.netgen_mesh.Curve(order)
        curvedPhysPts = np.ndarray((len(self.netgen_mesh.Elements3D()),
                                   refPts.shape[0], self.geometric_dimension()))
        self.netgen_mesh.CalcElementMapping(refPts, curvedPhysPts)
        cellMap = newFunctionCoordinates.cell_node_map()
        for i, el in enumerate(self.netgen_mesh.Elements3D()):
            if el.curved:
                pts = [tuple(map(rnd, pts))
                       for pts in physPts[i][0:refPts.shape[0]]]
                dofMap = {k: v for v, k in enumerate(pts)}
                p = [dofMap[tuple(map(rnd, pts))]
                     for pts in V[cellMap.values[getIdx(i)]][0:refPts.shape[0]]]
                curvedPhysPts[i] = curvedPhysPts[i][p]
                for j, datIdx in enumerate(cellMap.values[getIdx(i)][0:refPts.shape[0]]):
                    newFunctionCoordinates.sub(0).dat.data[datIdx] = curvedPhysPts[i][j][0]
                    newFunctionCoordinates.sub(1).dat.data[datIdx] = curvedPhysPts[i][j][1]
                    newFunctionCoordinates.sub(2).dat.data[datIdx] = curvedPhysPts[i][j][2]
    return newFunctionCoordinates

class FiredrakeMesh:
    '''
    This class creates a Firedrake mesh from Netgen/NGSolve meshes.

    :arg mesh: the mesh object, it can be either a Netgen/NGSolve mesh or a PETSc DMPlex
    :param netgen_flags: The dictionary of flags to be passed to ngsPETSc.
    :arg comm: the MPI communicator.
    '''
    def __init__(self, mesh, netgen_flags, user_comm=PETSc.COMM_WORLD):
        self.comm = user_comm
        if isinstance(mesh,(ngs.comp.Mesh,ngm.Mesh)):
            try:
                if netgen_flags["purify_to_tets"]:
                    mesh.Split2Tets()
            except KeyError:
                warnings.warn("No purify_to_tets flag found, mesh will not be purified to tets.")
            self.meshMap = MeshMapping(mesh)
        else:
            raise ValueError("Mesh format not recognised.")
        try:
            if netgen_flags["quad"]:
                transform = PETSc.DMPlexTransform().create(comm=PETSc.COMM_WORLD)
                transform.setType(PETSc.DMPlexTransformType.REFINETOBOX)
                transform.setDM(self.meshMap.petscPlex)
                transform.setUp()
                newplex = transform.apply(self.meshMap.petscPlex)
                self.meshMap = MeshMapping(newplex)
        except KeyError:
            warnings.warn("No quad flag found, mesh will not be quadrilateralised.")
        try:
            if netgen_flags["transform"] is not None:
                transform = netgen_flags["transform"]
                transform.setDM(self.meshMap.petscPlex)
                transform.setUp()
                newplex = transform.apply(self.meshMap.petscPlex)
                self.meshMap = MeshMapping(newplex)
        except KeyError:
            warnings.warn("No PETSc transform found, mesh will not be transformed.")

    def createFromTopology(self, topology, name):
        '''
        Internal method to construct a mesh from a mesh topology, copied from Firedrake.

        :arg topology: the mesh topology

        :arg name: the mesh name

        '''
        cell = topology.ufl_cell()
        geometric_dim = topology.topology_dm.getCoordinateDim()
        cell = cell.reconstruct(geometric_dimension=geometric_dim)
        element = ufl.VectorElement("Lagrange", cell, 1)
        # Create mesh object
        self.firedrakeMesh = fd.MeshGeometry.__new__(fd.MeshGeometry, element)
        self.firedrakeMesh._init_topology(topology)
        self.firedrakeMesh.name = name
        # Adding Netgen mesh and inverse sfBC as attributes
        self.firedrakeMesh.netgen_mesh = self.meshMap.ngMesh
        if self.comm.Get_size() > 1:
            self.firedrakeMesh.sfBCInv = self.firedrakeMesh.sfBC.createInverse()
        else:
            self.firedrakeMesh.sfBCInv = None
        self.firedrakeMesh.comm = self.comm
        setattr(fd.MeshGeometry, "refine_marked_elements", refineMarkedElements)
        setattr(fd.MeshGeometry, "curve_field", curveField)


class NetgenHierarchy(mg.HierarchyBase):
    
    def refineMarkedElements(fdMesh, mark):
        '''
        This method is used to refine a mesh based on a marking function
        which is a Firedrake DG0 function.

        :arg mark: the marking function which is a Firedrake DG0 function.

        '''
        
        if fdMesh.geometric_dimension() == 2:
            with mark.dat.vec as marked:
                marked0 = marked
                getIdx = fdMesh._cell_numbering.getOffset
                if fdMesh.sfBCInv is not None:
                    getIdx = lambda x: x
                    _, marked0 = fdMesh.topology_dm.distributeField(fdMesh.sfBCInv,
                                                                fdMesh._cell_numbering,
                                                                marked)
                if fdMesh.comm.Get_rank() == 0:
                    mark = marked0.getArray()
                    for i, el in enumerate(fdMesh.netgen_mesh.Elements2D()):
                        if mark[getIdx(i)]:
                            el.refine = True
                        else:
                            el.refine = False
                    fdMesh.netgen_mesh.Refine(adaptive=True)
                    return fd.Mesh(fdMesh.netgen_mesh)
                return (fd.Mesh(netgen.libngpy._meshing.Mesh(2)),0)
        else:
            raise NotImplementedError("No implementation for dimension other than 2.")
