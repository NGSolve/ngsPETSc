'''
This module contains all the functions related to the PETSc SNES
'''
import numpy as np

from petsc4py import PETSc

from ngsolve import GridFunction

from ngsPETSc import VectorMapping


class NonLinearSolver:
    '''
    This class creates a PETSc Krylov Solver (SNES) from a callback to
    a NGSolve residual vector
    '''
    def __init__(self, fes, a=None, residual=None, jacobian=None, solverParameters=None, optionsPrefix=None):
        self.fes = fes
        dofs = fes.ParallelDofs()
        self.snes = PETSc.SNES().create(comm=dofs.comm.mpi4py)
        #Setting up the options
        options_object = PETSc.Options()
        if solverParameters is not None:
            for optName, optValue in solverParameters.items():
                options_object[optName] = optValue
        self.snes.setOptionsPrefix(optionsPrefix)
        self.snes.setFromOptions()
        #Setting up utility for mappings
        self.vectorMapping = VectorMapping(self.fes)   
        if a is not None:
            def residual(x):
                res = GridFunction(fes)
                a.Apply(x.vec, res.vec)
                return res
            self.residual = residual
        elif residual is not None: self.residual = residual
        
        if a is not None:
            pass
        elif jacobian is not None: self.jacobian = jacobian
    def setup(self, x0):
        ngsGridFucntion = GridFunction(self.fes)
        pVec = self.vectorMapping.petscVec(ngsGridFucntion.vec)
        self.snes.setFunction(self.petscResidual, pVec)
        self.snes.view()
        self.pvec0 = self.vectorMapping.petscVec(x0.vec)
    def solve(self, x0):
        self.setup(x0)
        self.snes.solve(None,self.pvec0)
        self.solutionGridFucntion = GridFunction(self.fes)
        self.vectorMapping.ngsVec(self.pvec0,ngsVec=self.solutionGridFucntion.vec)
        return self.solutionGridFucntion
    def petscResidual(self,snes,x,f):
        ngsGridFuction = GridFunction(self.fes)
        self.vectorMapping.ngsVec(x,ngsVec=ngsGridFuction.vec)
        ngsGridFuction = self.residual(ngsGridFuction)
        self.vectorMapping.petscVec(ngsGridFuction.vec, petscVec=f)
        
    def residual(x):
        raise NotImplementedError("No residual has been implemented yet.")
        return x
    
    def petscJacobian(self,snes,x,J,P):
        ngsGridFuction = GridFunction(self.fes)
        ngsGridFuction.vec = self.vectorMapping.ngsVec(x)
    
    def jacobian(self,x):
        raise NotImplementedError("No Jacobian has been implemented yet.")
        return x