"""
This module wraps the PETSc TS time-stepping routines.
"""

from petsc4py import PETSc
from ngsolve import GridFunction, BilinearForm
from ngsPETSc import VectorMapping, Matrix

class TimeStepper:

    def __init__(self, fes, info, G=None, F=None, residual=None, jacobian=None, solverParameters=None, optionsPrefix=None):
        self.fes = fes
        dofs = fes.ParallelDofs()
        self.second_order = False
        self.trial = info[0]
        self.t = info[1]
        self.F = F
        self.G = G
        if "ngs_jacobian_mat_type" not in solverParameters:
            solverParameters["ngs_jacobian_mat_type"] = "aij"
        jacobianMatType = solverParameters["ngs_jacobian_mat_type"]
        self.ts = PETSc.TS().create(comm=dofs.comm.mpi4py)
        #Deafult TS setup
        self.ts.setExactFinalTime(2)
        self.ts.setMaxSNESFailures(-1)
        #Setting up the options
        options_object = PETSc.Options()
        if solverParameters is not None:
            for optName, optValue in solverParameters.items():
                options_object[optName] = optValue
        self.ts.setOptionsPrefix(optionsPrefix)
        self.ts.setFromOptions()
        #Setting up utility for mappings
        self.vectorMapping = VectorMapping(self.fes)
        if G is not None:
            def rhs(t, x): #pylint: disable=E0102, E0213
                self.t.Set(t)
                f = GridFunction(self.fes)
                a = BilinearForm(self.fes)
                a += self.G
                a.Assemble()
                a.Apply(x.vec, f.vec)
                return f
            self.rhs = rhs
        if F is not None:
            def residual(t, x, xdot): #pylint: disable=E0102, E0213
                res = GridFunction(self.fes)
                self.t.Set(t)
                updatedF = F.Replace({self.trial.dt: xdot})
                a = BilinearForm(self.fes)
                a += updatedF
                a.Assemble()
                a.Apply(x.vec, res.vec)
                return  res
            self.residual = residual
        else:
            raise ValueError("You need to provide a residual function or a variational form F.")
        if jacobian is not None:
            self.jacobian = jacobian
            self.jacobianMatType = jacobianMatType
            self.second_order = True
        elif F is not None:
            def jacobian(x, t, xdot): #pylint: disable=E0102,E0213
                self.t.Set(t)
                updatedF = F.Replace({self.trial.dt: xdot})
                a = BilinearForm(self.fes)
                a += updatedF
                a.AssembleLinearization(x.vec)
                return a.mat
            self.jacobian = jacobian
            self.second_order = True
            self.jacobianMatType = jacobianMatType
        else:
            raise ValueError("You need to provide a jacobian function or a variational form F.")
    def setup(self, timeInfo):
        '''
        This is method is used to setup the PETSc TS object
        '''
        if isinstance(timeInfo, (list,tuple)) and len(timeInfo) == 3:
            self.t0, self.tf, self.dt = timeInfo
        elif isinstance(timeInfo, dict):
            self.t0 = timeInfo["t0"]
            self.tf = timeInfo["tf"]
            self.dt = timeInfo["dt"]
        else:
            raise ValueError("timeInfo must be a list/tuple or a dict")
        ngsGridFucntion = GridFunction(self.fes)
        pIVec = self.vectorMapping.petscVec(ngsGridFucntion.vec)
        pEVec = self.vectorMapping.petscVec(ngsGridFucntion.vec)
        if self.G is not None:
            self.ts.setRHSFunction(self.petscRhs, pEVec)
        if self.F is not None:
            self.ts.setIFunction(self.petscResidual, pIVec)
        self.ts.setTime(self.t0)
        self.ts.setTimeStep(self.dt)
        self.ts.setMaxTime(self.tf)
        self.ts.setMaxSteps(int((self.tf-self.t0)/self.dt)+1)
        
    def solve(self, x0):
        pscx0 = self.vectorMapping.petscVec(x0.vec)
        self.ts.solve(pscx0)
        self.vectorMapping.ngsVec(pscx0, ngsVec=x0.vec)


    def petscRhs(self,ts,t,x,f):
        '''
        This is method is used to wrap the callback to the resiudal in
        a PETSc compatible way

        :arg ts: PETSc SNES object reppresenting the non-linear solver
        
        :arg x: current guess of the solution as a PETSc Vec

        :arg f: residual function as PETSc Vec
        '''
        assert isinstance(ts,PETSc.TS)
        ngsGridFuction = GridFunction(self.fes)
        self.vectorMapping.ngsVec(x,ngsVec=ngsGridFuction.vec)
        ngsGridFuction = self.rhs(t, ngsGridFuction)
        self.vectorMapping.petscVec(ngsGridFuction.vec, petscVec=f)

    def rhs(t, x): #pylint: disable=E0102,E0213,E0202
        '''
        Callback to the residual of the non-linear problem
        
        :arg x: current guess of the solution as a PETSc Vec

        :return: the residual as an NGSolve grid function
        '''
        raise NotImplementedError("No residual has been implemented yet.")

    def petscResidual(self,ts,t,x,xdot,f):
        '''
        This is method is used to wrap the callback to the resiudal in
        a PETSc compatible way

        :arg ts: PETSc SNES object reppresenting the non-linear solver
        
        :arg x: current guess of the solution as a PETSc Vec

        :arg f: residual function as PETSc Vec
        '''
        assert isinstance(ts,PETSc.TS)
        ngsGridFuction = GridFunction(self.fes)
        ngsGridFuctionDot = GridFunction(self.fes)
        self.vectorMapping.ngsVec(x,ngsVec=ngsGridFuction.vec)
        self.vectorMapping.ngsVec(xdot,ngsVec=ngsGridFuctionDot.vec)
        ngsGridFuction = self.residual(t, ngsGridFuction, ngsGridFuctionDot)
        self.vectorMapping.petscVec(ngsGridFuction.vec, petscVec=f)

    def residual(t, x): #pylint: disable=E0102,E0213,E0202
        '''
        Callback to the residual of the non-linear problem
        
        :arg x: current guess of the solution as a PETSc Vec

        :return: the residual as an NGSolve grid function
        '''
        raise NotImplementedError("No residual has been implemented yet.")

         
    def petscJacobian(self, ts, t, x, xdot, a, J,P):
        '''
        This is method is used to wrap the callback to the Jacobian in
        a PETSc compatible way

        :arg snes: PETSc SNES object reppresenting the non-linear solver
        
        :arg x: current guess of the solution as a PETSc Vec

        :arg J: Jacobian computed at x as a PETSc Mat

        :arg P: preconditioner for the Jacobian computed at x
                as a PETSc Mat
        '''
        assert isinstance(ts,PETSc.TS)
        ngsGridFuction = GridFunction(self.fes)
        self.vectorMapping.ngsVec(x, ngsVec=ngsGridFuction.vec)
        mat = self.jacobian(ngsGridFuction, t, xdot)
        Matrix(mat,self.fes, petscMat=P, matType=self.jacobianMatType)
        Matrix(mat,self.fes, petscMat=J, matType=self.jacobianMatType)

    def jacobian(x): #pylint: disable=E0102,E0213,E0202
        '''
        Callback to the Jacobian of the non-linear problem
        
        :arg x: current guess of the solution as a PETSc Vec

        :return: the Jacobian as an NGSolve matrix
        '''
        raise NotImplementedError("No Jacobian has been implemented yet.")
