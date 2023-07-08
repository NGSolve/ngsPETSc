'''
This module contains all the functions related to the SLEPc eigenvalue
solver (EPS/PEP) interface for NGSolve
'''
from petsc4py import PETSc
try:
    from slepc4py import SLEPc
except ImportError:
    import warnings
    warnings.warn("Import Warning: it was not possible to import SLEPc")
    SLEPc = None

from mpi4py import MPI

from ngsolve import GridFunction

from ngsPETSc import Matrix, VectorMapping
class EigenSolver():
    """
    This calss creates a SLEPc Eigen Problem Solver (EPS/PEP) from NGSolve
    variational problem pencil, i.e.
    a0(u,v)+lam*a1(u,v)+(lam^2)*a2(u,v)+ ... = 0
    Inspired by Firedrake Eigensolver class.

    :arg pencil: tuple containing the bilinear forms a: V x V -> K composing
    the pencil, e.g. (m,a) with a = BilinearForm(grad(u),grad(v)*dx) and
    m = BilinearForm(-1*u*v*dx)

    :arg fes: finite element space V

    :arg nev: number of requested eigenvalue

    :arg ncv: dimension of the internal subspace used by SLEPc,
    by Default by SLEPc.DECIDE

    :arg solverParameters: parameters to be passed to the KSP solver

    :arg optionsPrefix: special solver options prefix for this specific Krylov solver

    """
    if SLEPc is not None:
        def __init__(self, pencil, fes, nev, ncv=SLEPc.DECIDE, optionsPrefix=None,
                    solverParameters=None):
            self.comm = MPI.COMM_WORLD
            if not isinstance(pencil, tuple): pencil=tuple([pencil])
            self.penLength = len(pencil)
            self.fes = fes
            self.nev = nev
            self.ncv = ncv
            self.solverParameters = solverParameters
            self.optionsPrefix = optionsPrefix
            options_object = PETSc.Options()
            if solverParameters is not None:
                for optName, optValue in self.solverParameters.items():
                    options_object[optName] = optValue

            self.pencilMats = []
            self.pencilFlags = []
            for a in pencil:
                self.pencilFlags += [a.flags.ToDict()]
                self.pencilMats += [Matrix(a.Assemble().mat, fes.FreeDofs()).mat]
                self.pencilMats[-1].setOptionsPrefix(self.optionsPrefix)
                self.pencilMats[-1].setFromOptions()
            self.eps = None
            self.pep = None
            if self.penLength > 2:
                self.isEPS = False
            else:
                self.isEPS = True
                self.setUpEPS()

        def setUpEPS(self):
            '''
            This function setup a SLEPc EPS if the pencil has shape either (m,a)
            or is simply a single matrix. 
            '''
            self.eps = SLEPc.EPS().create()
            self.eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
            if self.penLength == 1:
                flag0 = self.pencilFlags[0]
                if "symmetric" in flag0.keys():
                    if flag0["symmetric"]:
                        self.eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
                    else:
                        self.eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
                else:
                    self.eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
                self.eps.setOperators(self.pencilMats[0])
            else:
                flag0 = self.pencilFlags[0]
                flag1 = self.pencilFlags[1]
                if "symmetric" in flag0.keys() and "symmetric" in flag1.keys():
                    if flag0["symmetric"] and flag1["symmetric"]:
                        self.eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
                    else:
                        self.eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
                else:
                    self.eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
                self.pencilMats[0].scale(-1)
                self.eps.setOperators(self.pencilMats[1], self.pencilMats[0])

            self.eps.setDimensions(self.nev, self.ncv)
            self.eps.setOptionsPrefix(self.optionsPrefix)
            self.eps.setFromOptions()

        def solve(self):
            '''
            This function solve the eigenprobelm
            '''
            self.eps.solve()
            self.nconv = self.eps.getConverged()
            if self.nconv == 0:
                raise RuntimeError("Did not converge any eigenvalues.")
            return self.nconv

        def view(self):
            '''
            This function setup display the information about SLEPc EPS/PEP
            '''
            self.eps.view()

        def eigenValue(self, i):
            '''
            This function return the eigenvalue of the eigenproblem

            :arg i: index of the eigenvalue we are intrested in.

            '''
            lam = None
            if self.isEPS:
                lam = self.eps.getEigenvalue(i)
            return lam

        def eigenFunction(self, i):
            '''
            This function return the eigenfunction of the eigenproblem

            :arg i: index of the eigenfunction we are intrested in.

            '''
            self.vecMap = VectorMapping(self.fes)
            eigenModeReal = GridFunction(self.fes)
            eigenModeImag = GridFunction(self.fes)
            eignModePETScReal = self.pencilMats[0].createVecLeft()
            eignModePETScImag = self.pencilMats[0].createVecLeft()
            if self.isEPS:
                self.eps.getEigenvector(i, eignModePETScReal, eignModePETScImag)
            self.vecMap.ngsVec(eignModePETScReal,eigenModeReal.vec)
            self.vecMap.ngsVec(eignModePETScImag,eigenModeImag.vec)
            return eigenModeReal, eigenModeImag

        def eigenValues(self, indeces):
            '''
            This function return the eigenvalues of the eigenproblem

            :arg indeces: indeces of the eigenvalues we are intrested in.

            '''
            lams = []
            if self.isEPS:
                for i in indeces:
                    lams = lams + [self.eps.getEigenvalue(i)]
            return lams

        def eigenFunctions(self, indeces):
            '''
            This function return a multidim with 
            the eigenfunctions of the eigenproblem

            :arg indeces: indeces of the eigenfunctions we are intrested in.

            '''
            self.vecMap = VectorMapping(self.fes)
            eigenModesReal = GridFunction(self.fes, multidim=self.nev)
            eigenModesImag = GridFunction(self.fes, multidim=self.nev)
            k = 0
            eignModePETScReal = self.pencilMats[0].createVecLeft()
            eignModePETScImag = self.pencilMats[0].createVecLeft()
            for i in indeces:
                if self.isEPS:
                    self.eps.getEigenvector(i, eignModePETScReal, eignModePETScImag)
                self.vecMap.ngsVec(eignModePETScReal,eigenModesReal.vecs[k])
                self.vecMap.ngsVec(eignModePETScImag,eigenModesImag.vecs[k])
                k += 1
            return eigenModesReal, eigenModesImag
