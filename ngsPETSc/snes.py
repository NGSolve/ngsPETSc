'''
This module contains all the functions related to the PETSc SNES
'''
from petsc4py import PETSc

from ngsolve import GridFunction

from ngsPETSc import VectorMapping, Matrix


class NonLinearSolver:
    '''
    This class creates a PETSc Non-Linear Solver (SNES) from a callback to
    a NGSolve residual vector

    :arg fes: the finite element space over which the non-linear problem
              is defined

    :arg a: the variational form reppresenting the non-linear problem

    :arg residual: callback to the residual for the non-linear solver,
                   this fuction is used only if the argument a is None.

    :arg objective: callback to the objective for the non-linear solver,
                   this fuction is used only if the argument a is None,
                   if False the PETSSc default is norm 2.

    :arg jacobian: callback to the Jacobian for the non-linear solver,
                   this fuction is used only if the argument a is None.
    '''
    def __init__(self, fes, a=None, residual=None, objective=None, jacobian=None,
                 solverParameters=None, optionsPrefix=None):
        self.fes = fes
        dofs = fes.ParallelDofs()
        self.second_order = False
        if "ngs_jacobian_mat_type" not in solverParameters:
            solverParameters["ngs_jacobian_mat_type"] = "aij"
        jacobianMatType = solverParameters["ngs_jacobian_mat_type"]
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
        if residual is not None: self.residual = residual
        elif a is not None:
            def residual(x):  #pylint: disable=E0102, E0213
                res = GridFunction(fes)
                a.Apply(x.vec, res.vec)
                return res
            self.residual = residual
        else:
            raise ValueError("Either evalFunction or a must be provided")
        if objective is not None: self.objective = objective
        elif a is not None:
            def objective(x):  #pylint: disable=E0102, E0213
                return a.Energy(x.vec)
            self.objective = objective
        if jacobian is not None:
            self.jacobian = jacobian
            self.jacobianMatType = jacobianMatType
            self.second_order = True
        elif a is not None:
            def jacobian(x): #pylint: disable=E0102,E0213
                a.AssembleLinearization(x.vec)
                return a.mat
            self.jacobian = jacobian
            self.second_order = True
            self.jacobianMatType = jacobianMatType
    def setup(self, x0):
        '''
        This is method is used to setup the PETSc SNES object

        :arg x0: NGSolve grid function reppresenting the initial guess
        '''
        ngsGridFucntion = GridFunction(self.fes)
        pVec = self.vectorMapping.petscVec(ngsGridFucntion.vec)
        self.snes.setFunction(self.petscResidual, pVec)
        if self.objective is not False: self.snes.setObjective(self.petscObjective)
        if self.second_order:
            J, P, _ = self.snes.getJacobian()
            self.snes.setJacobian(self.petscJacobian, J, P)
        self.pvec0 = self.vectorMapping.petscVec(x0.vec)
    def solve(self, x0):
        '''
        This is method solves the non-linear problem

        :arg x0: NGSolve grid function reppresenting the initial guess
        '''
        self.setup(x0)
        self.snes.solve(None,self.pvec0)
        self.solutionGridFucntion = GridFunction(self.fes)
        self.vectorMapping.ngsVec(self.pvec0,ngsVec=self.solutionGridFucntion.vec)
        return self.solutionGridFucntion
    def petscResidual(self,snes,x,f):
        '''
        This is method is used to wrap the callback to the resiudal in
        a PETSc compatible way

        :arg snes: PETSc SNES object reppresenting the non-linear solver
        
        :arg x: current guess of the solution as a PETSc Vec

        :arg f: residual function as PETSc Vec
        '''
        assert isinstance(snes,PETSc.SNES)
        ngsGridFuction = GridFunction(self.fes)
        self.vectorMapping.ngsVec(x,ngsVec=ngsGridFuction.vec)
        ngsGridFuction = self.residual(ngsGridFuction)
        self.vectorMapping.petscVec(ngsGridFuction.vec, petscVec=f)

    def residual(x): #pylint: disable=E0102,E0213,E0202
        '''
        Callback to the residual of the non-linear problem
        
        :arg x: current guess of the solution as a PETSc Vec

        :return: the residual as an NGSolve grid function
        '''
        raise NotImplementedError("No residual has been implemented yet.")

    def petscObjective(self, snes,x):
        '''
        This is method is used to wrap the callback to the objetcive in
        a PETSc compatible way

        :arg snes: PETSc SNES object reppresenting the non-linear solver
        
        :arg x: current guess of the solution as a PETSc Vec

        :arg energy: energy as a PETSc Scalar
        '''
        assert isinstance(snes,PETSc.SNES)
        ngsGridFuction = GridFunction(self.fes)
        self.vectorMapping.ngsVec(x,ngsVec=ngsGridFuction.vec)
        return self.objective(ngsGridFuction)

    def objective(x): #pylint: disable=E0102,E0213,E0202
        '''
        Callback to the objective of the non-linear problem
        
        :arg x: current guess of the solution as a PETSc Vec

        :return: the energy
        '''
        raise NotImplementedError("No residual has been implemented yet.")

    def petscJacobian(self,snes,x,J,P):
        '''
        This is method is used to wrap the callback to the Jacobian in
        a PETSc compatible way

        :arg snes: PETSc SNES object reppresenting the non-linear solver
        
        :arg x: current guess of the solution as a PETSc Vec

        :arg J: Jacobian computed at x as a PETSc Mat

        :arg P: preconditioner for the Jacobian computed at x
                as a PETSc Mat
        '''
        assert isinstance(snes,PETSc.SNES)
        ngsGridFuction = GridFunction(self.fes)
        self.vectorMapping.ngsVec(x, ngsVec=ngsGridFuction.vec)
        mat = self.jacobian(ngsGridFuction)
        Matrix(mat,self.fes, petscMat=P, matType=self.jacobianMatType)
        Matrix(mat,self.fes, petscMat=J, matType=self.jacobianMatType)

    def jacobian(x): #pylint: disable=E0102,E0213,E0202
        '''
        Callback to the Jacobian of the non-linear problem
        
        :arg x: current guess of the solution as a PETSc Vec

        :return: the Jacobian as an NGSolve matrix
        '''
        raise NotImplementedError("No Jacobian has been implemented yet.")
