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

#Firedrake webgui visualization
try:
    import firedrake
    import webgui_jupyter_widgets
except ImportError:
    firedrake = None
    webgui_jupyter_widgets = None

if firedrake and webgui_jupyter_widgets:
    from ngsPETSc.utils.firedrake_webgui import Draw, FiredrakeScene
    __all__ = __all__ + ["Draw", "FiredrakeScene"]

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
