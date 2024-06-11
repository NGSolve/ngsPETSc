'''
This module contains all the functions related to the PETSc linear
system solver (KSP) interface for NGSolve
'''
from petsc4py import PETSc

from ngsolve import la, BilinearForm, FESpace, BitArray

from ngsPETSc import Matrix, VectorMapping

def createFromBilinearForm(a, freeDofs, solverParameters, optionsPrefix):
    """
    This function creates a PETSc matrix from an NGSolve bilinear form
    """
    a.Assemble()
    #Setting deafult matrix type
    if "mat_type" not in solverParameters:
        solverParameters["mat_type"] = "aij"
    #Assembling matrix if not of type Python
    if solverParameters["mat_type"] not in ["python"]:
        if hasattr(a.mat, "row_pardofs"):
            dofs = a.mat.row_pardofs
        else:
            dofs = None
        mat = Matrix(a.mat, (dofs, freeDofs, None), solverParameters["mat_type"])
    return (a.mat, mat.mat)

class KrylovSolver():
    """
    This class creates a PETSc Krylov Solver (KSP) from NGSolve
    variational problem, i.e. a(u,v) = (f,v)
    Inspired by Firedrake linear solver class.

    :arg a: either the bilinear form, ngs Matrix or a petsc4py matrix

    :arg dofsDescr: either finite element space

    :arg p: either the bilinear form, ngs Matrix or petsc4py matrix actin as a preconditioner

    :arg solverParameters: parameters to be passed to the KS P solver

    :arg optionsPrefix: special solver options prefix for this specific Krylov solver

    """
    def __init__(self, a, dofsDescr=None, p=None, solverParameters=None, optionsPrefix=None, nullspace=None):
        # Grabbing parallel information
        if isinstance(dofsDescr, FESpace):
            freeDofs = dofsDescr.FreeDofs()
        elif isinstance(dofsDescr, BitArray):
            freeDofs = dofsDescr
        else:
            raise ValueError("dofsDescr must be either FESpace or BitArray")
        parse = {BilinearForm: createFromBilinearForm}
        #Construct operator        
        for key in parse:
            if isinstance(a, key):
                ngsA, pscA = parse[key](a, freeDofs, solverParameters, optionsPrefix)
        if p is not None:
            for key in parse:
                if isinstance(p, key):
                    ngsP, pscP = parse[key](p, freeDofs, solverParameters, optionsPrefix)
        else:
            ngsP = ngsA; pscP = pscA
        #Construct vector mapping
        if hasattr(ngsA, "row_pardofs"):
            dofs = ngsA.row_pardofsù
        else:
            dofs = None
        self.mapping = VectorMapping((dofs,freeDofs,{"bsize":ngsA.local_mat.entrysizes}))
        #Fixing PETSc options
        options_object = PETSc.Options()
        if solverParameters is not None:
            for optName, optValue in solverParameters.items():
                options_object[optName] = optValue

        #Creating the PETSc Matrix
        pscA.setOptionsPrefix(optionsPrefix)
        pscA.setFromOptions()
        pscP.setOptionsPrefix(optionsPrefix)
        pscP.setFromOptions()

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
