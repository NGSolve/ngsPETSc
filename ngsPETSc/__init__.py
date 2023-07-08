'''
ngsPETSc is a NGSolve/Netgen interface to PETSc
'''
from ngsPETSc.mat import *
from ngsPETSc.vec import *
from ngsPETSc.plex import *
from ngsPETSc.ksp import *
from ngsPETSc.pc import *
from ngsPETSc.eps import *

VERSION = "0.0.2"

__all__ = ["Matrix","VectorMapping","MeshMapping","KrylovSolver","EigenSolver"]
