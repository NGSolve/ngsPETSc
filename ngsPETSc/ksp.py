'''
This module contains all the functions related to the PETSc linear
system solver (KSP) interface for NGSolve
'''
from petsc4py import PETSc
from mpi4py import MPI

from ngsolve import la, GridFunction

from ngsPETSc import Matrix, VectorMapping

class KrylovSolver():
    """
    This calss creates a PETSc Krylov Solver (KSP) from NGSolve
    variational problem, i.e. a(u,v) = (f,v)
    Inspired by Firedrake linear solver class.

    :arg a: bilinear form a: V x V -> K

    :arg fes: finite element space V

    :arg p: bilinear form to be used for preconditioning

    :arg solverParameters: parameters to be passed to the KSP solver

    :arg optionsPrefix: special solver options prefix for this specific Krylov solver

    """
    def __init__(self, a, fes, p=None, solverParameters=None,optionsPrefix=None):
        self.fes = fes
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
        if Pmat is not None and not isinstance(Pmat, la.SparseMatrixd):
            raise TypeError("Provided preconditioner is a '%s', not an la.SparseMatrixd"
                            % type(Pmat).__name__)

        self.solverParameters = solverParameters
        self.optionsPrefix = optionsPrefix
        options_object = PETSc.Options()
        if solverParameters is not None:
            for optName, optValue in self.solverParameters.items():
                options_object[optName] = optValue
	#Creating the PETSc Matrix
        Asc = Matrix(Amat, fes.FreeDofs()).mat
        self.A = Asc
        self.comm = MPI.COMM_WORLD
        #Setting up the preconditioner
        if Pmat is not None:
            Psc = Matrix(Pmat, fes.FreeDofs()).mat
            self.P = Psc
        else:
            self.P = Asc
        #Setting options prefix
        self.A.setOptionsPrefix(self.optionsPrefix)
        self.P.setOptionsPrefix(self.optionsPrefix)
        self.A.setFromOptions()
        self.P.setFromOptions()

        self.ksp = PETSc.KSP().create(comm=self.comm)

        # Operator setting must come after null space has been
        # applied
        self.ksp.setOperators(A=self.A, P=self.P)
        self.ksp.setOptionsPrefix(self.optionsPrefix)
        self.ksp.setFromOptions()
    def solve(self, f):
        '''
        This function solves the linear system using a PETSc KSP. 

        :arg f: the data of the linear system

        '''
        f.Assemble()
        u = GridFunction(self.fes)
        self.vecMap = VectorMapping(self.fes)
        upsc, fpsc = self.A.createVecs()
        self.vecMap.petscVec(f.vec, fpsc)
        self.ksp.solve(fpsc, upsc)
        self.vecMap.ngsVec(upsc, u.vec)
        return  u

    def view(self):
        '''
        This function display PETSc KSP info

        '''
        self.ksp.view()
