'''
This module contains all the functions related 
'''
try:
    import firedrake as fd
    from firedrake.cython import mgimpl as impl
    from firedrake.__future__ import interpolate
except ImportError:
    fd = None

from fractions import Fraction
import numpy as np
from petsc4py import PETSc

from netgen.meshing import MeshingParameters

from ngsPETSc.utils.firedrake.meshes import flagsUtils

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

def snapToCoarse(coarse, linear, degree):
    '''
    This function snaps the coordinates of a DMPlex mesh to the coordinates of a Netgen mesh.
    '''
    print(linear.coordinates.dat.data.shape)
    if coarse.geometric_dimension() == 2:
        low_order_element = linear.coordinates.function_space().ufl_element().sub_elements[0]
        element = low_order_element.reconstruct(degree=degree)
        space = fd.VectorFunctionSpace(linear, fd.BrokenElement(element))
        ho = fd.assemble(interpolate(linear.coordinates, space))
        eStart, eEnd = linear.topology_dm.getDepthStratum(1)
        pStart, _ = linear.topology_dm.getDepthStratum(0)
        nodes = linear.topology_dm.getCoordinatesLocal().getArray().reshape(-1, 2)
        for i, pt in enumerate(ho.dat.data):
            d = np.sum((coarse.coordinates.dat.data - pt)**2, axis=1)
            j = np.argmin(d)
            #check if points are on the boudnary
            for k in range(eStart, eEnd):
                cone = linear.topology_dm.getCone(k)
                A = np.array([[1, nodes[cone[0]-pStart][0], nodes[cone[0]-pStart][1]],
                              [1, nodes[cone[1]-pStart][0], nodes[cone[1]-pStart][1]],
                              [1, pt[0], pt[1]]])
                lb = linear.topology_dm.getLabelValue("Face Sets", k)
                if np.abs(np.linalg.det(A)) <1e-12 and lb != -1:
                    ho.dat.data[i] = coarse.coordinates.dat.data[j]
    else:
        raise NotImplementedError("Snapping to Netgen meshes is only implemented for 2D meshes.")
    return fd.Mesh(ho, comm=linear.comm, distribution_parameters=linear._distribution_parameters)

def uniformRefinementRoutine(ngmesh, cdm):
    '''
    Routing called inside of NetgenHierarchy to compute refined ngmesh and plex.
    '''
    #We refine the netgen mesh uniformly
    ngmesh.Refine(adaptive=False)
    #We refine the DMPlex mesh uniformly
    cdm.setRefinementUniform(True)
    rdm = cdm.refine()
    rdm.removeLabel("pyop2_core")
    rdm.removeLabel("pyop2_owned")
    rdm.removeLabel("pyop2_ghost")
    return (rdm, ngmesh)

def uniformMapRoutine(meshes):
    '''
    This function computes the coarse to fine and fine to coarse maps
    for a uniform mesh hierarchy.
    '''
    refinements_per_level = 1
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
    return (coarse_to_fine_cells, fine_to_coarse_cells)

def alfeldRefinementRoutine(ngmesh, cdm):
    '''
    Routing called inside of NetgenHierarchy to compute refined ngmesh and plex.
    '''
    #We refine the netgen mesh alfeld
    ngmesh.SplitAlfeld()
    #We refine the DMPlex mesh alfeld
    tr = PETSc.DMPlexTransform().create(comm=PETSc.COMM_WORLD)
    tr.setType(PETSc.DMPlexTransformType.REFINEREGULAR)
    tr.setDM(cdm)
    tr.setUp()
    rdm = tr.apply(cdm)
    return (rdm, ngmesh)

def alfeldMapRoutine(meshes):
    '''
    This function computes the coarse to fine and fine to coarse maps
    for a alfeld mesh hierarchy.
    '''
    raise NotImplementedError("Alfeld refinement is not implemented yet.")

refinementTypes = {"uniform": (uniformRefinementRoutine, uniformMapRoutine),
                   "Alfeld": (alfeldRefinementRoutine, alfeldMapRoutine)}

def NetgenHierarchy(mesh, levs, flags):
    '''
    This function creates a Firedrake mesh hierarchy from Netgen/NGSolve meshes.

    :arg mesh: the Netgen/NGSolve mesh
    :arg levs: the number of levels in the hierarchy
    :arg netgen_flags: either a bool or a dictionray containing options for Netgen.
    If not False the hierachy is constructed using ngsPETSc, if None hierarchy
    constructed in a standard manner. Netgen flags includes:
        -degree, either an integer denoting the degree of curvature of all levels of
        the mesh or a list of levs+1 integers denoting the degree of curvature of
        each level of the mesh.
        -tol, geometric tollerance adopted in snapToNetgenDMPlex.
        -refinement_type, the refinment type to be used: uniform (default), Alfeld
    '''
    if mesh.geometric_dimension() == 3:
        raise NotImplementedError("Netgen hierachies are only implemented for 2D meshes.")
    ngmesh = mesh.netgen_mesh
    comm = mesh.comm
    #Parsing netgen flags
    if not isinstance(flags, dict):
        flags = {}
    order = flagsUtils(flags, "degree", 1)
    if isinstance(order, int):
        order= [order]*(levs+1)
    tol = flagsUtils(flags, "tol", 1e-8)
    refType = flagsUtils(flags, "refinement_type", "uniform")
    optMoves = flagsUtils(flags, "optimisation_moves", False)
    snap = flagsUtils(flags, "snap_to", "geometry")
    #Firedrake quoantities
    meshes = []
    coarse_to_fine_cells = []
    fine_to_coarse_cells = [None]
    params = {"partition": False}
    #We construct the unrefined linear mesh
    if mesh.comm.size > 1 and mesh._grown_halos:
        raise RuntimeError("Cannot refine parallel overlapped meshes ")
    #We curve the mesh
    if order[0]>1:
        mesh = fd.Mesh(mesh.curve_field(order=order[0], tol=tol),
                       distribution_parameters=params, comm=comm)
    meshes += [mesh]
    cdm = meshes[-1].topology_dm
    for l in range(levs):
        #Streightening the mesh
        ngmesh.Curve(1)
        rdm, ngmesh = refinementTypes[refType][0](ngmesh, cdm)
        cdm = rdm
        #We snap the mesh to the Netgen mesh
        if snap == "geometry":
            snapToNetgenDMPlex(ngmesh, rdm)
        #We construct a Firedrake mesh from the DMPlex mesh
        mesh = fd.Mesh(rdm, dim=meshes[-1].ufl_cell().geometric_dimension(), reorder=False,
                                    distribution_parameters=params, comm=comm)
        if optMoves:
            #Optimises the mesh, for example smoothing
            if ngmesh.dim == 2:
                ngmesh.OptimizeMesh2d(MeshingParameters(optimize2d=optMoves))
            elif mesh.dim == 3:
                ngmesh.OptimizeVolumeMesh(MeshingParameters(optimize3d=optMoves))
            else:
                raise ValueError("Only 2D and 3D meshes can be optimised.")
        mesh.netgen_mesh = ngmesh
        #We curve the mesh
        if order[l+1] > 1:
            if snap == "geometry":
                mesh = fd.Mesh(mesh.curve_field(order=order[l+1], tol=tol),
                               distribution_parameters=params, comm=comm)
            elif snap == "coarse":
                mesh = snapToCoarse(meshes[0], mesh, order[l+1])
        meshes += [mesh]
    #We populate the coarse to fine map
    coarse_to_fine_cells, fine_to_coarse_cells = refinementTypes[refType][1](meshes)
    return fd.HierarchyBase(meshes, coarse_to_fine_cells, fine_to_coarse_cells,
                            1, nested=False)
