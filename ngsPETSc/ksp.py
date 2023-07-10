'''
This module contains all the functions related to the PETSc linear
system solver (KSP) interface for NGSolve
'''
from petsc4py import PETSc

from ngsolve import la, GridFunction

from ngsPETSc import Matrix, VectorMapping

class KrylovSolver():
    """
    This class creates a PETSc Krylov Solver (KSP) from NGSolve
    variational problem, i.e. a(u,v) = (f,v)
    Inspired by Firedrake linear solver class.

    :arg a: bilinear form a: V x V -> K

    :arg fes: finite element space V

    :arg p: bilinear form to be used for preconditioning

    :arg solverParameters: parameters to be passed to the KSP solver

    :arg optionsPrefix: special solver options prefix for this specific Krylov solver

    """
    def __init__(self, a, fes, p=None, solverParameters=None, optionsPrefix=None):
        a.Assemble()
        Amat = a.mat
        if p is not None:
            p.Assemble()
            Pmat = p.mat
        else:
            Pmat = None
        if not isinstance(Amat, (la.SparseMatrixd,la.ParallelMatrix)):
            raise TypeError("Provided operator is a '%s', not an la.SparseMatrixd"
                            % type(Amat).__name__)
        if Pmat is not None and not isinstance(Pmat, la.SparseMatrixd, la.ParallelMatrix):
            raise TypeError("Provided preconditioner is a '%s', not an la.SparseMatrixd"
                            % type(Pmat).__name__)

        options_object = PETSc.Options()
        if solverParameters is not None:
            for optName, optValue in solverParameters.items():
                options_object[optName] = optValue

	#Creating the PETSc Matrix
        A = Matrix(Amat, fes.FreeDofs()).mat
        A.setOptionsPrefix(optionsPrefix)
        A.setFromOptions()
        P = A
        if Pmat is not None:
            P = Matrix(Pmat, fes.FreeDofs()).mat
            P.setOptionsPrefix(optionsPrefix)
            P.setFromOptions()

        self.ksp = PETSc.KSP().create(comm=A.getComm())
        self.ksp.setOperators(A=A, P=P)
        self.ksp.setOptionsPrefix(optionsPrefix)
        self.ksp.setFromOptions()

        self.upsc, self.fpsc = A.createVecs()

        self.vecMap = VectorMapping(fes)
        self.fes = fes

    def solve(self, f):
        '''
        This function solves the linear system using a PETSc KSP.

        :arg f: the data of the linear system

        '''
        f.Assemble()
        u = GridFunction(self.fes)
        self.vecMap.petscVec(f.vec, self.fpsc)
        self.ksp.solve(self.fpsc, self.upsc)
        self.vecMap.ngsVec(self.upsc, u.vec)
        return  u

    def view(self):
        '''
        This function display PETSc KSP info

        '''
        self.ksp.view()
