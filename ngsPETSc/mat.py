'''
This module contains all the functions related to wrapping NGSolve matrices to
PETSc matrices using the petsc4py interface.
'''
from petsc4py import PETSc
import numpy as np

class Mat(object):
    '''
    This calss creates a sparse PETSc Matrix

    :arg ngsMat: The NGSolve matrix

    :arg freeDofs: Free DOFs of the FE space used to construct the matrix

    :arg matType: Type of sparse matrix, i.e. PETSc, MKL or CUDA

    '''
    def __init__(self, ngsMat, freeDofs=None, matType="aij"):
        self.freeDofs = freeDofs
        self.comm = None
        localMat = ngsMat.local_mat
        entryHeight, entryWidth = localMat.entrysizes
        if entryHeight != entryWidth: raise RuntimeError ("Only square entries are allowed.")
        valMat,colMat,indMat = localMat.CSR()
        indMat = np.array(indMat).astype(PETSc.IntType)
        colMat = np.array(colMat).astype(PETSc.IntType)
        petscLocalMat = PETSc.Mat().createBAIJ(size=(entryHeight*localMat.height,
                                                     entryWidth*localMat.width),
                                               bsize=entryHeight,
                                               csr=(indMat,colMat,valMat),
                                               comm=self.comm)
        if self.freeDofs is not None:
            localMatFree = np.flatnonzero(self.freeDofs).astype(PETSc.IntType)
            isFreeLocal = PETSc.IS().createBlock(indices=localMatFree, bsize=entryHeight)
            petscLocalMat = petscLocalMat.createSubMatrices(isFreeLocal)[0]

        if self.comm.Get_size() > 1:
            matTypes = {"aij":"mpiaij","aijcusparse":"mpiaijcusparse","aijmkl":"mpiaijmkl"}
            parallelDofs = ngsMat.row_pardofs
            globalNums, numberGlobal = parallelDofs.EnumerateGlobally(self.freeDofs)
            if self.freeDofs is not None:
                globalNums = np.array(globalNums, dtype=PETSc.IntType)[self.freeDofs]
                localGlobalMap = PETSc.LGMap().create(indices=globalNums,
                                                      bsize=entryHeight,
                                                      comm=self.comm)
                mat = PETSc.Mat().create(comm=self.comm)
                mat.setSizes(size=numberGlobal*entryHeight, bsize=entryHeight)
                mat.setType(PETSc.Mat.Type.IS)
                mat.setLGMap(localGlobalMap)
                mat.setISLocalMat(petscLocalMat)
                mat.assemble()
                self.type = matTypes[matType]
                mat.convert(self.type)
                self.mat = mat
        else:
            if self.freeDofs is not None:
                mat = petscLocalMat
                mat.assemble()
                matTypes = {"aij":"seqaij","aijcusparse":"seqaijcusparse","aijmkl":"seqaijmkl"}
                self.type = matTypes[matType]
                mat.convert(self.type)
                self.mat = mat
