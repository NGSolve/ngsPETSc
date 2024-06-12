'''
This module contains all the functions related to the PETSc linear
system solver (KSP) interface for NGSolve
'''
from petsc4py import PETSc
from ngsolve import la, BilinearForm, FESpace, BitArray, Projector
from ngsPETSc import Matrix, VectorMapping, PETScPreconditioner

def createFromBilinearForm(a, freeDofs, solverParameters):
    """
    This function creates a PETSc matrix from an NGSolve bilinear form
    """
    a.Assemble()
    #Setting deafult matrix type
    if "ngs_mat_type" not in solverParameters:
        solverParameters["ngs_mat_type"] = "aij"
    #Assembling matrix if not of type Python
    if solverParameters["ngs_mat_type"] not in ["python"]:
        if hasattr(a.mat, "row_pardofs"):
            dofs = a.mat.row_pardofs
        else:
            dofs = None
        mat = Matrix(a.mat, (dofs, freeDofs, None), solverParameters["ngs_mat_type"])
    return (a.mat, mat.mat)

def createFromMatrix(a, freeDofs, solverParameters):
    """
    This function creates a PETSc matrix from an NGSolve bilinear form
    """
    #Setting deafult matrix type
    if "ngs_mat_type" not in solverParameters:
        solverParameters["ngs_mat_type"] = "aij"
    #Assembling matrix if not of type Python
    if solverParameters["ngs_mat_type"] not in ["python"]:
        if hasattr(a, "row_pardofs"):
            dofs = a.row_pardofs
        else:
            dofs = None
        mat = Matrix(a, (dofs, freeDofs, None), solverParameters["ngs_mat_type"])
    return (a, mat.mat)

def createFromPC(a, freeDofs, solverParameters):
    class Wrap(object):
        def __init__(self, a, freeDofs):
            if hasattr(a.ngsMat, "row_pardofs"):
                dofs = a.ngsMat.row_pardofs
            else:
                dofs = None
            self.a = a
            self.mapping = VectorMapping((dofs,freeDofs,{"bsize": [1]}))
            self.ngX = a.CreateColVector()
            self.ngY = a.CreateColVector()
            self.prj = Projector(mask=a.actingDofs, range=True)

        def mult(self, mat, X, Y):
            self.mapping.ngsVec(X, (self.prj*self.ngX).Evaluate())
            self.a.Mult(self.ngX, self.ngY)
            self.mapping.petscVec((self.prj*self.ngY).Evaluate(), Y)
        
    #Grabbing comm information
    if hasattr(a.ngsMat, "row_pardofs"):
        comm = a.ngsMat.row_pardofs.comm.mpi4py
    else:
        comm = PETSc.COMM_SELF
    pythonA = Wrap(a, freeDofs)
    pscA = PETSc.Mat().create(comm=comm)
    pscA.setSizes([sum(freeDofs), sum(freeDofs)])
    pscA.setType("python")
    pscA.setPythonContext(pythonA)
    pscA.setUp()
    return (a.ngsMat, pscA)

def createFromAction(a, freeDofs, solverParameters):
    class Wrap(object):
        def __init__(self, a, freeDofs):
            self.a = a
            self.mapping = VectorMapping((None,freeDofs,{"bsize": [1]}))
            self.ngX = a.CreateColVector()
            self.ngY = a.CreateColVector()

        def mult(self, mat, X, Y):
            self.mapping.ngsVec(X, self.ngX)
            self.a.Mult(self.ngX, self.ngY)
            self.mapping.petscVec(self.ngY, Y)
        
    pythonA = Wrap(a, freeDofs)
    pscA = PETSc.Mat().create(comm=PETSc.COMM_SELF)
    pscA.setSizes([sum(freeDofs), sum(freeDofs)])
    pscA.setType("python")
    pscA.setPythonContext(pythonA)
    pscA.setUp()
    return (a, pscA)

parse = {BilinearForm: createFromBilinearForm,
         la.SparseMatrixd: createFromMatrix,
         la.ParallelMatrix: createFromMatrix,
         PETScPreconditioner: createFromPC,
         la.BaseMatrix: createFromAction}

class KrylovSolver():
    """
    This class creates a PETSc Krylov Solver (KSP) for NGSolve.
    Inspired by Firedrake linear solver class.

    :arg a: either the bilinear form, ngs Matrix or a petsc4py matrix

    :arg dofsDescr: either finite element space

    :arg p: either the bilinear form, ngs Matrix or petsc4py matrix actin as a preconditioner

    :arg solverParameters: parameters to be passed to the KS P solver

    :arg optionsPrefix: special solver options prefix for this specific Krylov solver

    """
    def __init__(self, a, dofsDescr, p=None, nullspace=None, optionsPrefix="", solverParameters=None):
        # Grabbing dofs information
        if isinstance(dofsDescr, FESpace):
            freeDofs = dofsDescr.FreeDofs()
        elif isinstance(dofsDescr, BitArray):
            freeDofs = dofsDescr
        else:
            raise ValueError("dofsDescr must be either FESpace or BitArray")
        #Construct operator        
        for key in parse:
            if isinstance(a, key):
                ngsA, pscA = parse[key](a, freeDofs, solverParameters)
        if p is not None:
            for key in parse:
                if isinstance(p, key):
                    ngsP, pscP = parse[key](p, freeDofs, solverParameters)
                    break
        else:
            ngsP = ngsA; pscP = pscA
        
        #Construct vector mapping
        if hasattr(ngsA, "row_pardofs"):
            dofs = ngsA.row_pardofs
        else:
            dofs = None
        if hasattr(ngsA.local_mat, "entrysizes"):
            entrysize = ngsA.local_mat.entrysizes
        else:
            entrysize = [1]
        
        self.mapping = VectorMapping((dofs,freeDofs,{"bsize":entrysize}))
        #Fixing PETSc options
        options_object = PETSc.Options()
        if solverParameters is not None:
            for optName, optValue in solverParameters.items():
                options_object[optName] = optValue

        #Setting PETSc Options
        pscA.setOptionsPrefix(optionsPrefix)
        pscA.setFromOptions()

        #Setting up nullspace
        if nullspace is not None:
            pscA.setNullSpace(nullspace.nullspace)

        #Setting up KSP
        self.ksp = PETSc.KSP().create(comm=pscA.getComm())
        self.ksp.setOperators(A=pscA, P=pscP)
        self.ksp.setOptionsPrefix(optionsPrefix)
        self.ksp.setFromOptions()
        self.pscX, self.pscB = pscA.createVecs()

    def solve(self, b, x):
        """
        This function solves the linear system

        :arg b: right hand side of the linear system
        :arg x: solution of the linear system
        """
        self.mapping.petscVec(x, self.pscX)
        self.mapping.petscVec(b, self.pscB)
        self.ksp.solve(self.pscB, self.pscX)
        self.mapping.ngsVec(self.pscX, x)
