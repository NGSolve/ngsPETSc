'''
This module contains all the functions related to wrapping NGSolve matrices to
PETSc matrices using the petsc4py interface.
'''
import numpy as np

from ngsolve import FESpace
from petsc4py import PETSc

class Matrix(object):
    '''
    This class creates a PETSc Matrix

    :arg ngsMat: the NGSolve matrix

    :arg freeDofs: free DOFs of the FE spaces used to construct the matrix

    :arg matType: type of PETSc matrix, i.e. PETSc sparse: aij,
    MKL sparse: aijmkl or CUDA: aijcusparse

    '''
    def __init__(self, ngsMat, parDescr, matType="aij", petscMat=None):
        if not isinstance(parDescr, (tuple, list)):
            samerc = True
            if isinstance(parDescr, FESpace):
                dofs = parDescr.ParallelDofs()
                self.freeDofs = (parDescr.FreeDofs(),parDescr.FreeDofs())
                comm = dofs.comm.mpi4py
            else:
                dofs, freeDofs, dofsInfo = parDescr
                self.freeDofs = (freeDofs,freeDofs)
                if dofs is not None:
                    comm = dofs.comm.mpi4py
                else:
                    ### create suitable dofs
                    comm = PETSc.COMM_SELF
                    dofs = type('', (object,), {'entrysize':dofsInfo["bsize"][0]})()
        else:
            samerc = False
            if isinstance(parDescr[0], FESpace) and isinstance(parDescr[1], FESpace):
                dofs = parDescr[0].ParallelDofs()
                self.freeDofs = (parDescr[0].FreeDofs(),parDescr[1].FreeDofs())
                comm = dofs.comm.mpi4py
            elif isinstance(parDescr[0], (tuple, list)):
                dofs = parDescr[0][0]
                self.freeDofs = (parDescr[0][1], parDescr[1][1])
                if dofs is not None:
                    comm = dofs.comm.mpi4py
                else:
                    ### create suitable dofs
                    comm = PETSc.COMM_SELF
            else:
                samerc = True
                dofs = parDescr[0]
                self.freeDofs = (parDescr[1], parDescr[1])
                if dofs is not None:
                    comm = dofs.comm.mpi4py
                else:
                    ### create suitable dofs
                    comm = PETSc.COMM_SELF

        localMat = ngsMat.local_mat
        entryHeight, entryWidth = localMat.entrysizes
        if entryHeight != entryWidth: raise RuntimeError ("Only square entries are allowed.")

        valMat, colMat, indMat = localMat.CSR()
        indMat = np.array(indMat).astype(PETSc.IntType)
        colMat = np.array(colMat).astype(PETSc.IntType)
        if entryHeight > 1:
            petscLocalMat = PETSc.Mat().createBAIJ(size=(entryHeight*localMat.height,
                                                         entryWidth*localMat.width),
                                                   bsize=entryHeight,
                                                   csr=(indMat,colMat,valMat),
                                                   comm=PETSc.COMM_SELF)
        else:
            petscLocalMat = PETSc.Mat().createAIJ(size=(localMat.height,
                                                        localMat.width),
                                                  csr=(indMat,colMat,valMat),
                                                  comm=PETSc.COMM_SELF)
        if self.freeDofs[0] is not None or self.freeDofs[1] is not None:
            rowIsFreeLocal = None
            if self.freeDofs[0] is not None:
                rowLocalMatFree = np.flatnonzero(self.freeDofs[0]).astype(PETSc.IntType)
                rowIsFreeLocal = PETSc.IS().createBlock(indices=rowLocalMatFree,
                                                        bsize=entryHeight,comm=PETSc.COMM_SELF)
            colIsFreeLocal = rowIsFreeLocal
            if self.freeDofs[1] is not None and not samerc:
                colLocalMatFree = np.flatnonzero(self.freeDofs[1]).astype(PETSc.IntType)
                colIsFreeLocal = PETSc.IS().createBlock(indices=colLocalMatFree,
                                                        bsize=entryHeight,comm=PETSc.COMM_SELF)
            petscLocalMat = petscLocalMat.createSubMatrices(rowIsFreeLocal, colIsFreeLocal)[0]

        if comm.Get_size() > 1:
            # is this a BUG in the bindings?
            #rparallelDofs = ngsMat.row_pardofs
            rparallelDofs = ngsMat.col_pardofs
            rglobalNums, rnumberGlobal = rparallelDofs.EnumerateGlobally(self.freeDofs[0])
            if self.freeDofs[0] is not None:
                rglobalNums = np.array(rglobalNums, dtype=PETSc.IntType)[self.freeDofs[0]]
            rlocalGlobalMap = PETSc.LGMap().create(indices=rglobalNums,
                                                   bsize=entryHeight,
                                                   comm=comm)
            if not samerc:
                # is this a BUG in the bindings?
                #cparallelDofs = ngsMat.col_pardofs
                cparallelDofs = ngsMat.row_pardofs
                cglobalNums, cnumberGlobal = cparallelDofs.EnumerateGlobally(self.freeDofs[1])
                if self.freeDofs[1] is not None:
                    cglobalNums = np.array(cglobalNums, dtype=PETSc.IntType)[self.freeDofs[1]]
                clocalGlobalMap = PETSc.LGMap().create(indices=cglobalNums,
                                                       bsize=entryWidth,
                                                       comm=comm)
            else:
                clocalGlobalMap = rlocalGlobalMap
                cnumberGlobal = rnumberGlobal

            mat = PETSc.Mat().create(comm=comm)
            mat.setSizes(size=(rnumberGlobal*entryHeight,
                            cnumberGlobal*entryHeight), bsize=entryHeight)
            mat.setType(PETSc.Mat.Type.IS)
            mat.setLGMap(rlocalGlobalMap, clocalGlobalMap)
            mat.setISLocalMat(petscLocalMat)
            mat.assemble()
            if matType != 'is':
                mat.convert(matType)
            if petscMat is None:
                self.mat = mat
            else:
                mat.copy(petscMat)
        else:
            mat = petscLocalMat
            mat.convert(matType)

            if petscMat is None:
                self.mat = mat
            else:
                mat.copy(petscMat)

    def view(self):
        '''
        This function display PETSc Mat info

        '''
        self.mat.view()
