try:
    import firedrake as fd
    from petsc4py import PETSc
    from firedrake.__future__ import interpolate
except ImportError:
    fd = None
from ngsPETSc.plex import CELL_SETS_LABEL, FACE_SETS_LABEL
class TrefftzEmbedding(object):

    def __init__(self, V, b, dim=None, tol=1e-12):
        self.V = V
        self.b = b
        self.dim = V.dim() if not dim else dim + 1
        self.tol = tol
    
    def assemble(self, backend="PETSc"):
        self.B = fd.assemble(self.b).M.handle
        if backend == "scipy":
            import scipy.sparse as sp
            indptr, indices, data = self.B.getValuesCSR()
            Bsp = sp.csr_matrix((data, indices, indptr), shape=self.B.getSize())
            _, sig, VT = sp.linalg.svds(Bsp, k=self.dim-1, which="SM")
            QT = sp.csr_matrix(VT[0:sum(sig<self.tol), :])
            QTpsc = PETSc.Mat().createAIJ(size=QT.shape, csr=(QT.indptr, QT.indices, QT.data))
            self.dimT = QT.shape[0]
            return QTpsc, sig
        
    def assembledEmbeddedMatrix(self, a, backend="PETSc"):
        self.A = fd.assemble(a).M.handle
        self.QT, _ = self.assemble(backend)
        pythonQTAQ = self.embeddedMatrixWrap(self.QT, self.A)
        pscQTAQ = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        pscQTAQ.setSizes(self.dimT, self.dimT)
        pscQTAQ.setType("python")
        pscQTAQ.setPythonContext(pythonQTAQ)
        pscQTAQ.setUp()
        return pscQTAQ
    def assembledEmbeddedLoad(self, L):
        self.L = fd.assemble(L)
        with self.L.dat.vec as w:
            y =  self.QT.createVecLeft()
            self.QT.mult(w, y)
        return y
    def embed(self, y):
        u = fd.Function(self.V)
        with u.dat.vec as w:
            self.QT.multTranspose(y, w)
        return u

        
    class embeddedMatrixWrap(object):
        """
        This class wraps a PETSc Preconditioner as PETSc Python matrix
        """
        def __init__(self, QT, A):
            self.QT = QT
            self.A = A

        def mult(self, mat, X, Y): #pylint: disable=W0613
            """
            PETSc matrix-vector product
            """
            Z = self.QT.createVecRight()
            W = self.A.createVecRight()
            self.QT.multTranspose(X, Z)
            self.A.mult(Z, W)
            self.QT.mult(W, Y)

class AggregationEmbedding(TrefftzEmbedding):
    def __init__(self, V, mesh, dim=None, tol=1e-12):
        # Relabel facets that are inside an aggregated region
        plex = mesh.topology_dm
        pStart,pEnd = plex.getDepthStratum(2)
        numberBnd = len(mesh.netgen_mesh.GetRegionNames(dim=1))
        numberMat = len(mesh.netgen_mesh.GetRegionNames(dim=2))
        for mat in range(numberMat):
            facets = []
            for i in range(pStart,pEnd):
                if plex.getLabelValue(CELL_SETS_LABEL,i) == mat+1:
                    for f in plex.getCone(i):
                        if f in facets:
                            plex.setLabelValue(FACE_SETS_LABEL,f,numberBnd+numberMat+mat+1)
                    facets = facets + list(plex.getCone(i))
        self.mesh = fd.Mesh(plex)
        h = fd.CellDiameter(self.mesh)
        n = fd.FacetNormal(self.mesh)
        W = fd.FunctionSpace(self.mesh, V.ufl_element())
        u = fd.TrialFunction(W)
        v = fd.TestFunction(W)
        b = fd.Constant(0)*fd.inner(u,v)*fd.dx
        for i in range(numberBnd+numberMat+1, numberBnd+2*numberMat+1):
            b += fd.inner(fd.jump(u),fd.jump(v))*fd.dS(i)
        for k in range(V.ufl_element().degree()):
            for i in range(numberBnd+numberMat+1, numberBnd+2*numberMat+1):
                b += ((0.5*h("+")+0.5*h("-"))**(2*i+2))*fd.inner(jumpNormal(u,n("+")),jumpNormal(v, n("+")))*fd.dS(i)

        super().__init__(W, b, dim, tol)

def jumpNormal(u,n):
    return 0.5*fd.dot(n, (fd.grad(u)("+")-fd.grad(u)("-")))