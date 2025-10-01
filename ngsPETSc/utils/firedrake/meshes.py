'''
This module contains all the functions related to wrapping NGSolve meshes to Firedrake
We adopt the same docstring conventions as the Firedrake project, since this part of
the package will only be used in combination with Firedrake.
'''
import numpy as np
from mpi4py import MPI
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
from ngsPETSc.utils.utils import find_permutation

def flagsUtils(flags, option, default):
    '''
    utility fuction used to parse Netgen flag options
    '''
    try:
        return flags[option]
    except KeyError:
        return default


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
    def __init__(self, mesh, netgen_flags, user_comm=MPI.COMM_WORLD):
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
        elif isinstance(mesh, PETSc.DMPlex):
            self.meshMap = MeshMapping(mesh)
        else:
            raise ValueError("Mesh format not recognised.")
