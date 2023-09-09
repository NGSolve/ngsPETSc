'''
This module contains all the functions related to wrapping NGSolve meshes to Firedrake
'''
try:
    import firedrake as fd
except ImportError:
    fd = None

from ngsPETSc import MeshMapping
from petsc4py import PETSc
import ngsolve as ngs
import netgen.meshing as ngm
import ufl
import numpy as np

def refineMarkedElements(self, mark):
    if self.geometric_dimension() == 2:
        with mark.dat.vec as marked:
            marked0 = marked
            getIdx = self._cell_numbering.getOffset
            if self.sfBCInv is not None:
                getIdx = lambda x: x
                _, marked0 = self.topology_dm.distributeField(self.sfBCInv,
                                                              self._cell_numbering,
                                                              marked)
            if self.comm.Get_rank() == 0:
                mark = marked0.getArray()
                for i, el in enumerate(self.netgen_mesh.Elements2D()):
                    if mark[getIdx(i)]:
                        el.refine = True
                    else:
                        el.refine = False
                self.netgen_mesh.Refine(adaptive=True)
                return fd.Mesh(self.netgen_mesh)
            else:
                return fd.Mesh(netgen.libngpy._meshing.Mesh(2))
    else:
        raise NotImplementedError("No implementation for dimension other than 2.")

def curveField(self, order):
    functionCoordinates = self.coordinates
    newFunctionCoordinates = fd.interpolate(functionCoordinates,
                                            fd.VectorFunctionSpace(self,"DG",order))
    ref_element = functionCoordinates.function_space().finat_element.fiat_equivalent.ref_element
    getIdx = self._cell_numbering.getOffset
    if self.sfBCInv is not None:
        getIdx = lambda x: x
        _, marked0 = self.topology_dm.distributeField(self.sfBCInv,
                                                      self._cell_numbering,
                                                      marked)
    try:
        refPts = []
        for i in range(self.geometric_dimension()+1):
            for j in range(len(ref_element.sub_entities)):
                refPts = refPts+list(ref_element.make_points(i,j,order))
        print(refPts)
        refPts = np.array(refPts)
    except KeyError:
        raise NotImplementedError("The degrees of fredom you are looking for haven't been tabulated yet.")
    physPts = np.ndarray((len(self.netgen_mesh.Elements2D()), refPts.shape[0], 2))
    self.netgen_mesh.CalcElementMapping(refPts, physPts)
    cellMap = newFunctionCoordinates.cell_node_map()
    print(physPts)
    for i, el in enumerate(self.netgen_mesh.Elements2D()):
        for j, datIdx in enumerate(cellMap.values[getIdx(i)]):
            newFunctionCoordinates.sub(0).dat.data[datIdx] = physPts[i][j][0]
            newFunctionCoordinates.sub(1).dat.data[datIdx] = physPts[i][j][1]
    return newFunctionCoordinates

class FiredrakeMesh:
    '''
    This class creates a Firedrake mesh from Netgen/NGSolve meshes.

    :arg mesh: the mesh object, it can be either a Netgen/NGSolve mesh or a PETSc DMPlex
    :arg name: the name of to be assigned to the PETSc DMPlex, by default this is set to "Default"
    '''
    def __init__(self, mesh, user_comm=PETSc.COMM_WORLD):
        self.comm = user_comm
        if isinstance(mesh,(ngs.comp.Mesh,ngm.Mesh)):
            self.meshMap = MeshMapping(mesh)
        else:
            raise ValueError("Mesh format not recognised.")
    def createFromTopology(self, topology, name):
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
        self.firedrakeMesh.sfBCInv = mesh.sfBC.createInverse() if self.comm.Get_size() > 1 else None
        self.firedrakeMesh.comm = self.comm
        setattr(fd.MeshGeometry, "refine_marked_elements", refineMarkedElements)
        setattr(fd.MeshGeometry, "curve_field", curveField)
