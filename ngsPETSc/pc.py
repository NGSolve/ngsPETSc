'''
This module contains all the function and class needed to wrap a PETSc Preconditioner in NGSolve
'''
from petsc4py import PETSc
from mpi4py import MPI

from ngsolve import BaseMatrix

from ngsPETSc import Matrix, VectorMapping

class PETScPreconditioner(BaseMatrix):
    def __init__(self,mat,freeDofs, solverParameters=None, optionsPrefix=None, matType="aij"):
        BaseMatrix.__init__(self)
        self.ngsMat = mat
        if MPI.COMM_WORLD.Get_size() > 1:
            self.dofs = self.ngsMat.row_pardofs 
            self.freeDofs = freeDofs
        else:
            raise RuntimeError("PETSc PC implemented only in parallel.")
        self.vecMap = VectorMapping (None,parDofs=self.dofs,freeDofs=self.freeDofs)
        self.petscMat = Matrix(self.ngsMat, freeDofs, matType).mat
        self.petscPreconditioner = PETSc.PC().create()
        self.petscPreconditioner.setOperators(self.petscMat)
        self.petscPreconditioner.setFromOptions()
        self.solverParameters = solverParameters.ToDict()
        self.optionsPrefix = optionsPrefix
        options_object = PETSc.Options()
        if solverParameters is not None:
            for optName, optValue in self.solverParameters.items():
                options_object[optName] = optValue

        self.petscPreconditioner.setUp()
        self.petscVecX, self.petscVecY = self.petscMat.createVecs()

    def Shape(self):
        return self.ngsMat.shape

    def CreateVector(self,col):
        return self.ngsMat.CreateVector(not col)
    
    def Mult(self,x,y):
        self.vecMap.petscVec(x,self.petscVecX)
        self.petscPreconditioner.apply(self.petscVecX, self.petscVecY)
        self.vecMap.ngsVec(self.petscVecY, y)
        
    def MultTrans(self,x,y):
        self.vecMap.petscVec(x,self.petscVecX)
        self.petscPreconditioner.applyTranspose(self.petscVecX, self.petscVecY)
        self.vecMap.ngsVec(self.petscVecY, y)
        

def createPETScPreconditioner(mat, freeDofs, solverParameters):
    return PETScPreconditioner(mat, freeDofs, solverParameters)


from ngsolve.comp import RegisterPreconditioner
RegisterPreconditioner ("PETScPC", createPETScPreconditioner)
