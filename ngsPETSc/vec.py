'''
This module contains all the functions related to the NGSolve vector -
PETSc vector mapping using the petsc4py interface.
'''
import numpy as np

from petsc4py import PETSc

from ngsolve import la, FESpace

class VectorMapping:
    '''
    This class creates a mapping between a PETSc vector and NGSolve
    vectors

    :arg parDescr: the finite element space for the vector or tuple (dofs, freeDofs)
    :arg prefix: prefix for PETSc options

    '''
    def __init__(self, parDescr, prefix='ngs_'):
        if isinstance(parDescr, FESpace):
            dofs = parDescr.ParallelDofs()
            freeDofs = parDescr.FreeDofs()
            comm = dofs.comm.mpi4py
        else:
            dofs, freeDofs, dofsInfo = parDescr
            if dofs is not None:
                comm = dofs.comm.mpi4py
            else:
                ### create suitable dofs
                comm = PETSc.COMM_WORLD
                dofs = type('', (object,), {'entrysize':dofsInfo["bsize"][0]})()

        self.dofs = dofs
        bsize = dofs.entrysize
        locfree = np.flatnonzero(freeDofs).astype(PETSc.IntType)
        self.isetlocfree = PETSc.IS().createBlock(indices=locfree,
                                             bsize=bsize, comm=PETSc.COMM_SELF)
        nloc = len(freeDofs)

        nglob = len(locfree)
        if comm.Get_size() > 1:
            globnums, nglob = dofs.EnumerateGlobally(freeDofs)
            globnums = np.array(globnums, dtype=PETSc.IntType)[freeDofs]
            self.iset = PETSc.IS().createBlock(indices=globnums, bsize=bsize, comm=comm)
        else:
            self.iset = PETSc.IS().createBlock(indices=np.arange(nglob,dtype=PETSc.IntType),
                                          bsize=bsize, comm=comm)

        self.sVec = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        self.sVec.setSizes(nloc*bsize,bsize=bsize)
        self.sVec.setOptionsPrefix(prefix)
        self.sVec.setFromOptions()

        self.pVec = PETSc.Vec().create(comm=comm)
        self.pVec.setSizes(nglob*bsize,bsize=bsize)
        self.pVec.setBlockSize(bsize)
        self.pVec.setOptionsPrefix(prefix)
        self.pVec.setFromOptions()

        self.ngsToPETScScat = PETSc.Scatter().create(self.sVec, self.isetlocfree,
                                                     self.pVec, self.iset)

    def petscVec(self, ngsVec, petscVec=None):
        '''
        This function generate a PETSc vector from a NGSolve vector

        :arg ngsVec: the NGSolve vector
        :arg petscVec: the PETSc vector to be loaded with NGSolve
        vector, if None new PETSc vector is generated, by deafault None.

        '''
        if petscVec is None:
            petscVec = self.pVec.duplicate()
        ngsVec.Distribute()
        self.sVec.placeArray(ngsVec.FV().NumPy())
        petscVec.set(0)
        self.ngsToPETScScat.scatter(self.sVec, petscVec, addv=PETSc.InsertMode.ADD,
                                    mode=PETSc.ScatterMode.FORWARD)
        self.sVec.resetArray()
        return petscVec

    def ngsVec(self, petscVec, ngsVec=None):
        '''
        This function generate a NGSolve vector from a PETSc vector

        :arg petscVec: the PETSc vector
        :arg ngsVec: the NGSolve vector vector to be loaded with PETSc
        vector, if None new PETSc vector is generated, by deafault None.

        '''
        if ngsVec is None:
            ngsVec = la.CreateParallelVector(self.dofs,la.PARALLEL_STATUS.CUMULATED)
        ngsVec[:] = 0
        self.sVec.placeArray(ngsVec.FV().NumPy())
        self.PETScToNGSScat = PETSc.Scatter().create(petscVec, self.iset,
                                                     self.sVec, self.isetlocfree)
        self.PETScToNGSScat.scatter(petscVec, self.sVec, addv=PETSc.InsertMode.INSERT)
        self.sVec.resetArray()
        return ngsVec
