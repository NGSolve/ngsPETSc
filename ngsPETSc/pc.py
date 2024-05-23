'''
This module contains all the function and class needed to wrap a PETSc Preconditioner in NGSolve
'''
from petsc4py import PETSc

from ngsolve import BaseMatrix, comp
from ngsPETSc import Matrix, VectorMapping



class PETScPreconditioner(BaseMatrix):
    '''
    This class creates a Netgen/NGSolve BaseMatrix corresponding to a PETSc PC  

    :arg mat: NGSolve Matrix one would like to build the PETSc preconditioner for.

    :arg freeDofs: not constrained degrees of freedom of the finite element space over
    which the BilinearForm corresponding to the matrix is defined.

    :arg solverParameters: parameters to be passed to the KSP solver

    :arg optionsPrefix: special solver options prefix for this specific Krylov solver

    :arg matType: type of sparse matrix, i.e. PETSc sparse: aij, 
    MKL sparse: mklaij or CUDA: aijcusparse

    '''
    nullsapce = None
    def __init__(self, mat, freeDofs, solverParameters=None, optionsPrefix=None):
        BaseMatrix.__init__(self),
        matType="aij"
        if hasattr(solverParameters, "ToDict"):
            solverParameters = solverParameters.ToDict()
        if "restrictedTo" in solverParameters:
            freeDofs = solverParameters["restrictedTo"]
        if "matType" in solverParameters:
            matType = solverParameters["matType"]
        self.ngsMat = mat
        if hasattr(self.ngsMat, "row_pardofs"):
            dofs = self.ngsMat.row_pardofs
        else:
            dofs = None
        self.vecMap = VectorMapping((dofs,freeDofs,{"bsize":self.ngsMat.local_mat.entrysizes}))
        petscMat = Matrix(self.ngsMat, (dofs, freeDofs, None), matType).mat
        self.petscPreconditioner = PETSc.PC().create(comm=petscMat.getComm())
        self.petscPreconditioner.setOperators(petscMat)
        options_object = PETSc.Options()
        if solverParameters is not None:
            for optName, optValue in solverParameters.items():
                if optName not in ["restrictedTo", "matType"]:
                    options_object[optName] = optValue
        self.petscPreconditioner.setOptionsPrefix(optionsPrefix)
        self.petscPreconditioner.setFromOptions()
        self.petscPreconditioner.setUp()
        self.petscVecX, self.petscVecY = petscMat.createVecs()

    def Shape(self):
        '''
        Shape of the BaseMatrix

        '''
        return self.ngsMat.shape

    def CreateVector(self,col):
        '''
        Create vector corresponding to the matrix

        :arg col: True if one want a column vector

        '''
        return self.ngsMat.CreateVector(not col)

    def Mult(self,x,y):
        '''
        BaseMatrix multiplication Ax = y
        :arg x: vector we are multiplying
        :arg y: vector we are storeing the result in

        '''
        self.vecMap.petscVec(x,self.petscVecX)
        self.petscPreconditioner.apply(self.petscVecX, self.petscVecY)
        self.vecMap.ngsVec(self.petscVecY, y)

    def MultTrans(self,x,y):
        '''
        BaseMatrix multiplication A^T x = y
        :arg x: vector we are multiplying
        :arg y: vector we are storeing the result in

        '''
        self.vecMap.petscVec(x,self.petscVecX)
        self.petscPreconditioner.applyTranspose(self.petscVecX, self.petscVecY)
        self.vecMap.ngsVec(self.petscVecY, y)

def createPETScPreconditioner(mat, freeDofs, solverParameters):
    '''
    Create PETSc PC that can be accessed by NGSolve.
    :arg mat: NGSolve Matrix one would like to build the PETSc preconditioner for.

    :arg freeDofs: not constrained degrees of freedom of the finite element space over
    which the BilinearForm corresponding to the matrix is defined.

    :arg solverParameters: parameters to be passed to the KSP solver

    '''
    return PETScPreconditioner(mat, freeDofs, solverParameters)


comp.RegisterPreconditioner ("PETScPC", createPETScPreconditioner)
