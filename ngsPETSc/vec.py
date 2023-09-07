'''
This module contains all the functions related to the NGSolve vector - 
PETSc vector mapping using the petsc4py interface.
'''
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from ngsolve import la

class VectorMapping:
    '''
    This calss creates a mapping between a PETSc vector and NGSolve
    vectors

    :arg fes: the finite element space for the vector we want to map

    '''
    def __init__(self, fes, parDofs=None, freeDofs=None, comm=MPI.COMM_WORLD):
        if fes is not None:
            self.fes = fes
            self.dofs = self.fes.ParallelDofs()
            self.freeDofs = self.fes.FreeDofs()
        else:
            self.dofs = parDofs
            self.freeDofs = freeDofs
        self.comm = comm
        print("dofs",self.dofs)
        if comm.Get_size() > 1:
            globnums, self.nglob = self.dofs.EnumerateGlobally(self.freeDofs)
            self.es = self.dofs.entrysize
            if self.freeDofs is not None:
                globnums = np.array(globnums, dtype=PETSc.IntType)[self.freeDofs]
                self.locfree = np.flatnonzero(self.freeDofs).astype(PETSc.IntType)
                self.isetlocfree = PETSc.IS().createBlock(indices=self.locfree,
                                                          bsize=self.es, comm=comm)
            else:
                self.isetlocfree = None
                globnums = list(range(len(self.freeDofs)))
            self.iset = PETSc.IS().createBlock(indices=globnums, bsize=self.es, comm=comm)
            self.isetlocfree.view()
            self.iset.view()
        

    def petscVec(self, ngsVec, petscVec=None):
        '''
        This function generate a PETSc vector from a NGSolve vector

        :arg ngsVec: the NGSolve vector
        :arg petscVec: the PETSc vector to be loaded with NGSolve
        vector, if None new PETSc vector is generated, by deafault None.

        '''
        ngsVec.Distribute()
        if self.comm.Get_size() > 1:
            if petscVec is None:
                petscVec = PETSc.Vec().createMPI(self.nglob*self.es, bsize=self.es, comm=self.comm)
            locvec = PETSc.Vec().createWithArray(ngsVec.FV().NumPy(), comm=MPI.COMM_SELF)
            if "ngsTopetscScat" not in self.__dict__:
                self.ngsToPETScScat = PETSc.Scatter().create(locvec, self.isetlocfree,
                                                            petscVec, self.iset)
            petscVec.set(0)
            self.ngsToPETScScat.scatter(locvec, petscVec, addv=PETSc.InsertMode.ADD)
        else:
            if petscVec is None:
                petscVec = PETSc.Vec().createWithArray(ngsVec.FV().NumPy(), comm=self.comm)
            else:
                freeIndeces = np.flatnonzero(self.freeDofs).astype(PETSc.IntType)
                petscVec.set(0)
                petscVec.setArray(ngsVec.FV().NumPy()[freeIndeces])

        return petscVec

    def ngsVec(self, petscVec, ngsVec=None):
        '''
        This function generate a NGSolve vector from a PETSc vector

        :arg petscVec: the PETSc vector
        :arg ngsVec: the NGSolve vector vector to be loaded with PETSc
        vector, if None new PETSc vector is generated, by deafault None.

        '''
        print("PETSc Vec", petscVec.size)
        if self.comm.Get_size() > 1:
            if ngsVec is None:
                ngsVec = la.CreateParallelVector(self.dofs,la.PARALLEL_STATUS.CUMULATED)
            ngsVec[:] = 0.0
            locvec = PETSc.Vec().createWithArray(ngsVec.FV().NumPy(), comm=MPI.COMM_SELF)
            if "petscToNgsScat" not in self.__dict__:
                self.petscToNgsScat = PETSc.Scatter().create(petscVec, self.iset,
                                                             locvec, self.isetlocfree)
            self.petscToNgsScat.scatter(petscVec, locvec, addv=PETSc.InsertMode.INSERT)
        else:
            if ngsVec is None:
                ngsVec = la.BaseVector(len(self.freeDofs),False) #Only work for real vector
            ngsVec[:] = 0.0
            freeIndeces = np.flatnonzero(self.freeDofs).astype(PETSc.IntType)
            ngsVec.FV().NumPy()[freeIndeces] = petscVec.getArray()
        return ngsVec
