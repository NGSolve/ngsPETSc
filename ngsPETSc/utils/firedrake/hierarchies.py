'''
This module contains all the functions related 
'''
try:
    import firedrake as fd
    from firedrake.cython import mgimpl as impl
    from firedrake.__future__ import interpolate
    import firedrake.dmhooks as dmhooks
    import ufl
except ImportError:
    fd = None

from fractions import Fraction
import numpy as np
from scipy import optimize
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
    dim = coarse.geometric_dimension()
    if dim == 2:
        coarseSpace = coarse.coordinates.function_space()
        space = fd.VectorFunctionSpace(linear, "CG", degree)
        ho = fd.assemble(interpolate(coarse.coordinates, space))
        bnd =space.boundary_nodes("on_boundary")
        coarseBnd = coarseSpace.boundary_nodes("on_boundary")
        cellMap = coarse.coordinates.cell_node_map()
        for i in bnd:
            pt = ho.dat.data[i]
            j = np.argmin(np.sum((coarse.coordinates.dat.data[coarseBnd]- pt)**2, axis=1))
            print(np.linalg.norm(pt-coarse.coordinates.dat.data[coarseBnd[j]]))
            if np.linalg.norm(pt-coarse.coordinates.dat.data[coarseBnd[j]]) > 1e-16:
                cell_idx = coarse.locate_cell(pt)
                dim = linear.cell_sizes.at(pt)
                edge_nodes = [node for node in cellMap.values[cell_idx] if node in coarseBnd]
                #Compute polynomial reppresentation of the cell edge
                edge_coo = coarse.coordinates.dat.data[edge_nodes]
                coeffs = np.polyfit(edge_coo[:, 0], edge_coo[:, 1], degree)
                dalet = lambda x: np.sqrt((x-pt[0])**2 + (np.polyval(coeffs, x)-pt[1])**2)
                newpt = optimize.minimize_scalar(dalet)
                ho.dat.data[i] = (newpt.x, np.polyval(coeffs, newpt.x))
        """
        #Hyperelastic Smoothing
        bcs = [fd.DirichletBC(space, ho, "on_boundary")]
        quad_degree = 2*(degree+1)-1
        dx = fd.dx(degree=quad_degree, domain=linear)
        d = linear.topological_dimension()

        Q = fd.TensorFunctionSpace(linear, "DG", degree=0)
        Jinv = ufl.JacobianInverse(linear)
        hinv = fd.Function(Q)
        hinv.interpolate(Jinv)
        G = ufl.Jacobian(linear) * hinv
        ijac = 1/abs(ufl.det(G))
        ref_grad = lambda u: ufl.dot(ufl.grad(u), G)
        params = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "l2",
            "snes_max_it": 50,
            "snes_rtol": 1E-8,
            "snes_atol": 1E-8,
            "snes_ksp_ew": True,
            "snes_ksp_ew_rtol0": 1E-2,
            "snes_ksp_ew_rtol_max": 1E-2,
        }
        params["mat_type"] = "aij"
        coarse = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_mat_factor_type": "mumps",
        }
        gmg = {
            "pc_type": "mg",
            "mg_coarse": coarse,
            "mg_levels": {
                "ksp_max_it": 2,
                "ksp_type": "chebyshev",
                "pc_type": "jacobi",
            },
        }
        l = fd.mg.utils.get_level(linear)[1]
        pc = gmg if l else coarse
        params.update(pc)
        ksp = {
            "ksp_rtol": 1E-8,
            "ksp_atol": 0,
            "ksp_type": "minres",
            "ksp_norm_type": "preconditioned",
        }
        params.update(ksp)
        u = ho
        F = ref_grad(u)
        J = ufl.det(F)
        psi = (1/2) * (ufl.inner(F, F)-d - ufl.ln(J**2))
        U = (psi * ijac)*fd.dx(degree=quad_degree)
        dU = ufl.derivative(U, u, fd.TestFunction(space))
        problem = fd.NonlinearVariationalProblem(dU, u, bcs)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params)
        solver.set_transfer_manager(None)
        ctx = solver._ctx
        for c in problem.F.coefficients():
            dm = c.function_space().dm
            dmhooks.push_appctx(dm, ctx)
        solver.solve()
        """
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
        CG = True if snap == "coarse" else False
        mesh = fd.Mesh(mesh.curve_field(order=order[0], tol=tol, CG=CG),
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
