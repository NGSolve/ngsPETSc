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

    :arg a: the variational form representing the non-linear problem

    :arg objective: If True, adds a objective function callback to SNES
                    If False the PETSc default is norm 2.

    :arg jacobian: If True, adds a Jacobian callback to SNES
    '''
    def __init__(self, fes, a=None, use_objective=False, use_jacobian=True,
                 solverParameters={}, optionsPrefix=""):
        self.fes = fes
        dofs = fes.ParallelDofs()
        self.snes = PETSc.SNES().create(comm=dofs.comm.mpi4py)
        self.a = a
        #Setting up the options
        options_object = PETSc.Options(optionsPrefix)
        for optName, optValue in solverParameters.items():
            if optName not in options_object:
                options_object[optName] = optValue
        self.snes.setOptionsPrefix(optionsPrefix)
        if use_objective:
            self.snes.setObjective(self.petscObjective)
        self.use_jacobian = False
        if use_jacobian:
            self.snes.setJacobian(self.petscJacobian)
            self.use_jacobian = True
        self.jacobianMatType = solverParameters.get("ngs_jacobian_mat_type", "aij")
        #Setting up utility for mappings
        self.vectorMapping = VectorMapping(self.fes)
        self.workgf = (GridFunction(self.fes), GridFunction(self.fes))
        pVec = self.vectorMapping.petscVec()
        self.snes.setFunction(self.petscResidual, pVec)
        self.snes.setSolution(pVec.duplicate())

    def solve(self, x0 = None, f = None):
        '''
        This method solves the non-linear problem

        :arg x0: optional NGSolve grid function representing the initial guess
        :arg x0: optional NGSolve grid function representing the affine part of the rhs
        '''
        X = self.snes.getSolution()
        if x0:
            self.vectorMapping.petscVec(x0.vec, petscVec=X)
        else:
            x0 = GridFunction(self.fes)
        F = None
        if f:
            F = self.vectorMapping.petscVec(f.vec)
        if self.use_jacobian:
            # need to sample the Jacobian to give PETSc the matrix
            # object
            P = self.petscJacobian(self.snes, X, None, None)
            self.snes.setJacobian(None,J=P,P=P)
        self.snes.setFromOptions()
        self.snes.setUp()
        self.snes.solve(F, X)
        self.vectorMapping.ngsVec(X, ngsVec=x0.vec)
        return x0

    def petscResidual(self, _snes, x, f):
        '''
        This method is used to wrap the callback for the residual in
        a PETSc compatible way

        :arg snes: PETSc SNES object representing the non-linear solver

        :arg x: current guess of the solution as a PETSc Vec

        :arg f: residual function as PETSc Vec
        '''
        self.vectorMapping.ngsVec(x, ngsVec=self.workgf[0].vec)
        self.residual(self.workgf[0],self.workgf[1])
        self.vectorMapping.petscVec(self.workgf[1].vec, petscVec=f)

    def petscObjective(self, _snes, x):
        '''
        This method is used to wrap the callback for the objective in
        a PETSc compatible way

        :arg snes: PETSc SNES object representing the non-linear solver

        :arg x: current guess of the solution as a PETSc Vec
        '''
        self.vectorMapping.ngsVec(x, ngsVec=self.workgf[0].vec)
        obj = self.objective(self.workgf[0])
        return x.getComm().tompi4py().allreduce(obj)

    def petscJacobian(self, _snes, x, J, P):
        '''
        This method is used to wrap the callback for the Jacobian in
        a PETSc compatible way

        :arg snes: PETSc SNES object representing the non-linear solver

        :arg x: current guess of the solution as a PETSc Vec

        :arg J: Jacobian computed at x as a PETSc Mat

        :arg P: preconditioner for the Jacobian computed at x
                as a PETSc Mat
        '''
        self.vectorMapping.ngsVec(x, ngsVec=self.workgf[0].vec)
        mat = self.jacobian(self.workgf[0])
        ngsMat = Matrix(mat, self.fes, petscMat=P, matType=self.jacobianMatType)
        if P is None:
            return ngsMat.mat.duplicate()
        if J != P:
            J.assemble()
        return None

    def objective(self, x):
        '''
        Default callback for the objective of the non-linear problem

        :arg x: current iterate as an NGSolve grid function

        :return: the energy
        '''
        return self.a.Energy(x.vec)

    def residual(self, x, f):
        '''
        Default callback for the residual of the non-linear problem

        :arg x: current iterate as an NGSolve grid function

        :arg f: residual as as an NGSolve grid function

        '''
        self.a.Apply(x.vec, f.vec)

    def jacobian(self, x):
        '''
        Default callback for the Jacobian of the non-linear problem

        :arg x: current iterate as an NGSolve grid function

        :return: the Jacobian as an NGSolve matrix
        '''
        self.a.AssembleLinearization(x.vec)
        return self.a.mat
