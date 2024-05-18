'''
ngsPETSc is a NGSolve/Netgen interface to PETSc
'''
import warnings

from ngsPETSc.plex import *

#Firedrake
try:
    import firedrake
except ImportError:
    firedrake = None

if firedrake:
    from ngsPETSc.utils.firedrake import *

#FEniCSx
try:
    import dolfinx
except ImportError:
    dolfinx = None

if dolfinx:
    from ngsPETSc.utils.fenicsx import *

#Netgen
try:
    import ngsolve
except ImportError:
    warnings.warn("No NGSolve installed, only working with Netgen.")
    ngsolve = None

if ngsolve:
    from ngsPETSc.mat import *
    from ngsPETSc.vec import *
    from ngsPETSc.ksp import *
    from ngsPETSc.nullspace import *
    from ngsPETSc.pc import *
    from ngsPETSc.eps import *
    from ngsPETSc.snes import *

VERSION = "0.0.5"

__all__ = ["Matrix","VectorMapping","MeshMapping","KrylovSolver","EigenSolver",
           "FiredrakeMesh","NullSpace"]
