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
    def __init__(self, mat, freeDofs, solverParameters=None, optionsPrefix="", nullspace=None,
                 matType="aij", blocks=None):
        BaseMatrix.__init__(self)
        if hasattr(solverParameters, "ToDict"):
            solverParameters = solverParameters.ToDict()
        self.ngsMat = mat
        if hasattr(self.ngsMat, "row_pardofs"):
            dofs = self.ngsMat.row_pardofs
        else:
            dofs = None
        self.vecMap = VectorMapping((dofs,freeDofs,{"bsize":self.ngsMat.local_mat.entrysizes}))
        petscMat = Matrix(self.ngsMat, (dofs, freeDofs, None), matType).mat
        if nullspace is not None:
            if nullspace.near:
                petscMat.mat.setNearNullSpace(nullspace.nullspace)
        self.petscPreconditioner = PETSc.PC().create(comm=petscMat.getComm())
        self.petscPreconditioner.setOperators(petscMat)
        options_object = PETSc.Options()
        if solverParameters is not None:
            for optName, optValue in solverParameters.items():
                if "sub_pc_type"==optName and blocks is not None:
                    options_object["sub_pc_type"] = "lu"
                if "sub_pc_factor_mat_ordering_type"==optName and blocks is not None:
                    options_object["sub_pc_factor_mat_ordering_type"] = "natural"
                options_object[optName] = optValue
        self.petscPreconditioner.setOptionsPrefix(optionsPrefix)
        self.petscPreconditioner.setFromOptions()
        self.asmpc = None
        if blocks is not None:
            if len (blocks) == 0:
                self.ises = [PETSc.IS().createGeneral([0], comm=PETSc.COMM_SELF)]
            else:
                lgmap = petscMat.getLGMap()[0] #TODO: Fiox this only works for rc
                self.ises = [lgmap.applyIS(PETSc.IS().createGeneral(list(block),
                             comm=PETSc.COMM_SELF)) for block in blocks]
            self.asmpc = PETSc.PC().create(comm=self.petscPreconditioner.getComm())
            self.asmpc.incrementTabLevel(1, parent=self.petscPreconditioner)
            self.asmpc.setOptionsPrefix(optionsPrefix)
            self.asmpc.setFromOptions()
            self.asmpc.setOperators(*self.petscPreconditioner.getOperators())
            self.asmpc.setType(self.asmpc.Type.ASM)
            self.asmpc.setASMType(PETSc.PC.ASMType.BASIC)
            self.asmpc.setASMLocalSubdomains(len(self.ises), self.ises)
            self.asmpc.setUp()
        else:
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
        if self.asmpc is not None:
            self.asmpc.apply(self.petscVecX, self.petscVecY)
        else:
            self.petscPreconditioner.apply(self.petscVecX, self.petscVecY)
        self.vecMap.ngsVec(self.petscVecY, y)

    def MultTrans(self,x,y):
        '''
        BaseMatrix multiplication A^T x = y
        :arg x: vector we are multiplying
        :arg y: vector we are storeing the result in

        '''
        self.vecMap.petscVec(x,self.petscVecX)
        if self.asmpc is not None:
            self.asmpc.applyTranspose(self.petscVecX, self.petscVecY)
        else:
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
