'''
ngsPETSc is a NGSolve/Netgen interface to PETSc
'''
#initialize PETSc first
import sys
import petsc4py

petsc4py.init(sys.argv)

from ngsPETSc.plex import * #pylint: disable=C0413

__all__ = []

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
    ngsolve = None

if ngsolve:
    from ngsPETSc.mat import *
    from ngsPETSc.vec import *
    from ngsPETSc.nullspace import *
    from ngsPETSc.pc import *
    from ngsPETSc.ksp import *
    from ngsPETSc.snes import *
    from ngsPETSc.eps import *
    __all__ = __all__ + ["Matrix","VectorMapping","MeshMapping",
                         "KrylovSolver","EigenSolver","NullSpace",
                         "PETScPreconditioner", "NonLinearSolver"]
