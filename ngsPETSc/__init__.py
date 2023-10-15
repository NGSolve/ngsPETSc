'''
ngsPETSc is a NGSolve/Netgen interface to PETSc
'''
import warnings
from ngsPETSc.plex import *
from ngsPETSc.utils.firedrake import *
from ngsPETSc.utils.fenicsx import *
try:
    import ngsolve
except ImportError:
    warnings.warn("No NGSolve installed, only working with Netgen.")
    ngsolve = None
if ngsolve:
    from ngsPETSc.mat import *
    from ngsPETSc.vec import *
    from ngsPETSc.ksp import *
    from ngsPETSc.pc import *
    from ngsPETSc.eps import *
    from ngsPETSc.snes import *

VERSION = "0.0.3"

__all__ = ["Matrix","VectorMapping","MeshMapping","KrylovSolver","EigenSolver","FiredrakeMesh"]
