'''
This module contains all the functions related to wrapping NGSolve meshes to Firedrake
We adopt the same docstring conventiona as the Firedrake project, since this part of
the package will only be used in combination with Firedrake.
'''
try:
    import firedrake as fd
    from firedrake.logging import warning
    from firedrake.cython import mgimpl as impl
    from firedrake.__future__ import interpolate
except ImportError:
    fd = None

from fractions import Fraction
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
            return fd.Mesh(netgen.libngpy._meshing.Mesh(2))

    elif self.geometric_dimension() == 3:
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
                for i, el in enumerate(self.netgen_mesh.Elements3D()):
                    if mark[getIdx(i)]:
                        el.refine = True
                    else:
                        el.refine = False
                self.netgen_mesh.Refine(adaptive=True)
                return fd.Mesh(self.netgen_mesh)
            return fd.Mesh(netgen.libngpy._meshing.Mesh(3))
    else:
        raise NotImplementedError("No implementation for dimension other than 2 and 3.")

def curveField(self, order, tol=1e-8):
    '''
    This method returns a curved mesh as a Firedrake function.

    :arg order: the order of the curved mesh

    '''
    low_order_element = self.coordinates.function_space().ufl_element().sub_elements[0]
    element = low_order_element.reconstruct(degree=order)
    space = fd.VectorFunctionSpace(self, fd.BrokenElement(element))
    newFunctionCoordinates = fd.assemble(interpolate(self.coordinates, space))
    self.netgen_mesh = self.comm.bcast(self.netgen_mesh, root=0)
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
            #Inefficent code but runs only on curved elements
            if el.curved:
                pts = physPts[i][0:refPts.shape[0]]
                bary = sum([np.array(pts[i]) for i in range(len(pts))])/len(pts)
                Idx = self.locate_cell(bary)
                isInMesh = (0<=Idx<len(cellMap.values)) if Idx is not None else False
                if isInMesh:
                    p = [np.argmin(np.sum((pts - pt)**2, axis=1))
                         for pt in V[cellMap.values[Idx]][0:refPts.shape[0]]]
                    curvedPhysPts[i] = curvedPhysPts[i][p]
                    res = np.linalg.norm(pts[p]-V[cellMap.values[Idx]][0:refPts.shape[0]])
                    if res > tol:
                        fd.logging.warning("Not able to curve Firedrake element {}".format(Idx))
                    else:
                        for j, datIdx in enumerate(cellMap.values[Idx][0:refPts.shape[0]]):
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
            #Inefficent code but runs only on curved elements
            if el.curved:
                pts = physPts[i][0:refPts.shape[0]]
                bary = sum([np.array(pts[i]) for i in range(len(pts))])/len(pts)
                Idx = self.locate_cell(bary)
                isInMesh = (0<=Idx<len(cellMap.values)) if Idx is not None else False
                if isInMesh:
                    p = [np.argmin(np.sum((pts - pt)**2, axis=1))
                         for pt in V[cellMap.values[Idx]][0:refPts.shape[0]]]
                    curvedPhysPts[i] = curvedPhysPts[i][p]
                    res = np.linalg.norm(pts[p]-V[cellMap.values[Idx]][0:refPts.shape[0]])
                    if res > tol:
                        warning("Not able to curve element {}, residual is: {}".format(Idx, res))
                    else:
                        for j, datIdx in enumerate(cellMap.values[Idx][0:refPts.shape[0]]):
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
    def __init__(self, mesh, netgen_flags, user_comm=fd.COMM_WORLD):
        self.comm = user_comm
        if isinstance(mesh,(ngs.comp.Mesh,ngm.Mesh)):
            try:
                if netgen_flags["purify_to_tets"]:
                    mesh.Split2Tets()
            except KeyError:
                warnings.warn("No purify_to_tets flag found, mesh will not be purified to tets.")
            self.meshMap = MeshMapping(mesh, comm=self.comm)
        else:
            raise ValueError("Mesh format not recognised.")
        try:
            if netgen_flags["quad"]:
                transform = PETSc.DMPlexTransform().create(comm=self.comm)
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
        element = fd.VectorElement("Lagrange", cell, 1)
        # Create mesh object
        self.firedrakeMesh = fd.MeshGeometry.__new__(fd.MeshGeometry, element)
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

def snapToNetgenDMPlex(ngmesh, petscPlex):
    '''
    This function snaps the coordinates of a DMPlex mesh to the coordinates of a Netgen mesh.
    '''
    if petscPlex.getDimension() == 2:
        ngCoordinates = ngmesh.Coordinates()
        petscCoordinates = petscPlex.getCoordinatesLocal().getArray().reshape(-1, ngmesh.dim)
        for i, pt in enumerate(petscCoordinates):
            j = np.argmin(np.sum((ngCoordinates - pt)**2, axis=1))
            petscCoordinates[i] = ngCoordinates[j]
        petscPlexCoordinates = petscPlex.getCoordinatesLocal()
        petscPlexCoordinates.setArray(petscPlexCoordinates)
        petscPlex.setCoordinatesLocal(petscPlexCoordinates)
    else:
        raise NotImplementedError("Snapping to Netgen meshes is only implemented for 2D meshes.")

def flagsUtils(flags, option, default):
    '''
    utility fuction used to parse Netgen flag options
    '''
    try:
        return flags[option]
    except KeyError:
        return default

def NetgenHierarchy(mesh, levs, flags, tol=1e-8):
    '''
    This function creates a Firedrake mesh hierarchy from Netgen/NGSolve meshes.

    :arg mesh: the Netgen/NGSolve mesh
    :arg levs: the number of levels in the hierarchy
    :ar netgen_flags: either a bool or a dictionray containing options for Netgen.
    If not False the hierachy is constructed using ngsPETSc, if None hierarchy
    constructed in a standard manner.
    '''
    if mesh.geometric_dimension() == 3:
        raise NotImplementedError("Netgen hierachies are only implemented for 2D meshes.")
    ngmesh = mesh.netgen_mesh
    comm = mesh.comm
    #Parsing netgen flags
    if isinstance(flags, dict):
        order = flagsUtils(flags, "degree", 1)
    else:
        order = 1
    #Firedrake quoantities
    meshes = []
    refinements_per_level = 1
    coarse_to_fine_cells = []
    fine_to_coarse_cells = [None]
    params = {"partition": False}
    #We construct the unrefined linear mesh
    if mesh.comm.size > 1 and mesh._grown_halos:
        raise RuntimeError("Cannot refine parallel overlapped meshes ")
    #We curve the mesh
    if order>1:
        mesh = fd.Mesh(mesh.curve_field(order=order, tol=tol),
                       distribution_parameters=params, comm=comm)
    meshes += [mesh]
    cdm = meshes[-1].topology_dm
    for _ in range(levs):
        #Streightening the mesh
        ngmesh.Curve(1)
        #We refine the netgen mesh uniformly
        ngmesh.Refine(adaptive=False)
        #We refine the DMPlex mesh uniformly
        cdm.setRefinementUniform(True)
        rdm = cdm.refine()
        rdm.removeLabel("pyop2_core")
        rdm.removeLabel("pyop2_owned")
        rdm.removeLabel("pyop2_ghost")
        cdm = rdm
        #We snap the mesh to the Netgen mesh
        snapToNetgenDMPlex(ngmesh, rdm)
        #We construct a Firedrake mesh from the DMPlex mesh
        mesh = fd.Mesh(rdm, dim=meshes[-1].ufl_cell().geometric_dimension(), reorder=False,
                                    distribution_parameters=params, comm=comm)
        mesh.netgen_mesh = ngmesh
        #We curve the mesh
        if order > 1:
            mesh = fd.Mesh(mesh.curve_field(order=order, tol=1e-8),
                           distribution_parameters=params, comm=comm)
        meshes += [mesh]
    #We populate the coarse to fine map
    lgmaps = []
    for i, m in enumerate(meshes):
        no = impl.create_lgmap(m.topology_dm)
        m.init()
        o = impl.create_lgmap(m.topology_dm)
        m.topology_dm.setRefineLevel(i)
        lgmaps.append((no, o))
    coarse_to_fine_cells = []
    fine_to_coarse_cells = [None]
    for (coarse, fine), (clgmaps, flgmaps) in zip(zip(meshes[:-1], meshes[1:]),
                                                zip(lgmaps[:-1], lgmaps[1:])):
        c2f, f2c = impl.coarse_to_fine_cells(coarse, fine, clgmaps, flgmaps)
        coarse_to_fine_cells.append(c2f)
        fine_to_coarse_cells.append(f2c)

    coarse_to_fine_cells = dict((Fraction(i, refinements_per_level), c2f)
                                for i, c2f in enumerate(coarse_to_fine_cells))
    fine_to_coarse_cells = dict((Fraction(i, refinements_per_level), f2c)
                                for i, f2c in enumerate(fine_to_coarse_cells))
    return fd.HierarchyBase(meshes, coarse_to_fine_cells, fine_to_coarse_cells,
                         refinements_per_level, nested=False)
