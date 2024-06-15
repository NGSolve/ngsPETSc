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
    raise ValueError("ngs_mat_type {} is not supported.".format(solverParameters["ngs_mat_type"]))

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
        pscMat = mat.mat
    elif solverParameters["ngs_mat_type"] == "python":
        _, pscMat = createFromAction(a, freeDofs, solverParameters)
        return (a, pscMat)
    raise ValueError("ngs_mat_type {} is not supported.".format(solverParameters["ngs_mat_type"]))

def createFromPC(a, freeDofs, solverParameters):
    """
    This function creates a PETSc matrix from an ngsPETSc PETSc Preconditioner
    """
    class Wrap(object):
        """
        This class wraps a PETSc Preconditioner as PETSc Python matrix
        """
        def __init__(self, a, dofs, freeDofs):
            self.a = a
            self.mapping = VectorMapping((dofs,freeDofs,{"bsize": [1]}))
            self.ngX = a.CreateColVector()
            self.ngY = a.CreateColVector()
            self.prj = Projector(mask=a.actingDofs, range=True)

        def mult(self, mat, X, Y): #pylint: disable=W0613
            """
            PETSc matrix-vector product
            """
            self.mapping.ngsVec(X, (self.prj*self.ngX).Evaluate())
            self.a.Mult(self.ngX, self.ngY)
            self.mapping.petscVec((self.prj*self.ngY).Evaluate(), Y)
    #Grabbing comm information
    if hasattr(a, "row_pardofs"):
        dofs = a.row_pardofs
        comm = dofs.comm.mpi4py
    elif "dofs" in solverParameters:
        dofs = solverParameters["dofs"]
        comm = dofs.comm.mpi4py
    else:
        dofs = None
        comm = PETSc.COMM_SELF
    pythonA = Wrap(a, dofs, freeDofs)
    pscA = PETSc.Mat().create(comm=comm)
    pscA.setSizes([sum(freeDofs), sum(freeDofs)])
    pscA.setType("python")
    pscA.setPythonContext(pythonA)
    pscA.setUp()
    return (a.ngsMat, pscA)

def createFromAction(a, freeDofs, solverParameters):
    """
    This function creates a matrix free PETSc matrix from an NGSolve matrix
    """
    class Wrap(object):
        """
        This class wraps an NGSolve matrix as PETSc Python matrix
        """
        def __init__(self, a, dofs, freeDofs, comm):
            self.a = a
            self.dofs = dofs
            self.freeDofs = freeDofs
            self.comm = comm
            self.mapping = VectorMapping((dofs,freeDofs,{"bsize": (1,1)}))
            self.ngX = a.CreateColVector()
            self.ngY = a.CreateColVector()

        def mult(self, mat, X, Y): #pylint: disable=W0613
            """
            PETSc matrix-vector product
            """
            self.mapping.ngsVec(X, self.ngX)
            self.a.Mult(self.ngX, self.ngY)
            self.mapping.petscVec(self.ngY, Y)

    if hasattr(a, "row_pardofs"):
        dofs = a.row_pardofs
        comm = dofs.comm.mpi4py
        entrysize = a.local_mat.entrysizes[0]
        _, rnumberGlobal = dofs.EnumerateGlobally(freeDofs) #samrc
    elif "dofs" in solverParameters:
        dofs = solverParameters["dofs"]
        comm = dofs.comm.mpi4py
        entrysize = dofs.entrysize
        _, rnumberGlobal = dofs.EnumerateGlobally(freeDofs) #samrc
    else:
        dofs = None
        comm = PETSc.COMM_SELF
        entrysize = 1
        rnumberGlobal = sum(freeDofs)
    pythonA = Wrap(a, dofs, freeDofs, comm)
    pscA = PETSc.Mat().create(comm=comm)
    pscA.setSizes(size=(rnumberGlobal*entrysize,
                        rnumberGlobal*entrysize), bsize=entrysize)
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
    def __init__(self, a, dofsDescr, p=None, nullspace=None, optionsPrefix="",
                 solverParameters=None):
        # Grabbing dofs information
        if isinstance(dofsDescr, FESpace):
            freeDofs = dofsDescr.FreeDofs()
        elif isinstance(dofsDescr, BitArray):
            freeDofs = dofsDescr
        else:
            raise ValueError("dofsDescr must be either FESpace or BitArray")
        #Construct operator
        pscA = None
        for key in parse: #pylint: disable=C0206
            if isinstance(a, key):
                ngsA, pscA = parse[key](a, freeDofs, solverParameters)
        if pscA is None:
            raise ValueError("a of type {} not supported.".format(type(a)))
        if p is not None:
            for key in parse:  #pylint: disable=C0206
                if isinstance(p, key):
                    if hasattr(ngsA, "row_pardofs"):
                        solverParameters["dofs"] = ngsA.row_pardofs
                    _, pscP = parse[key](p, freeDofs, solverParameters)
                    break
        else:
            pscP = pscA
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
            if isinstance(nullspace, PETSc.NullSpace):
                pscA.setNullSpace(nullspace)
            else:
                pscA.setNullSpace(nullspace.nullspace)

        #Setting up KSP
        self.ksp = PETSc.KSP().create(comm=pscA.getComm())
        self.ksp.setOperators(A=pscA, P=pscP)
        self.ksp.setOptionsPrefix(optionsPrefix)
        self.ksp.setFromOptions()
        self.pscX, self.pscB = pscA.createVecs()

    def solve(self, b, x, mapping=None):
        """
        This function solves the linear system

        :arg b: right hand side of the linear system
        :arg x: solution of the linear system
        """
        if mapping is None:
            mapping = self.mapping
        mapping.petscVec(x, self.pscX)
        mapping.petscVec(b, self.pscB)
        self.ksp.solve(self.pscB, self.pscX)
        mapping.ngsVec(self.pscX, x)
