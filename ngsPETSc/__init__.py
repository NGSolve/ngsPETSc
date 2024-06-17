'''
ngsPETSc is a NGSolve/Netgen interface to PETSc
'''
from ngsPETSc.plex import *

__all__ = []

#Firedrake
try:
    import firedrake
except ImportError:
    firedrake = None

if firedrake:
    from ngsPETSc.utils.firedrake.meshes import *
    from ngsPETSc.utils.firedrake.hierarchies import *
    __all__ = __all__ + ["FiredrakeMesh", "NetgenHierarchy"]

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

VERSION = "0.0.5"
