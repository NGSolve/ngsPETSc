try:
    import firedrake as fd
    from petsc4py import PETSc
    from firedrake.__future__ import interpolate
except ImportError:
    fd = None

class TrefftzEmbedding(object):

    def __init__(self, V, b, dim=None, tol=1e-12):
        self.V = V
        self.b = b
        self.dim = V.dim() if not dim else dim
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
            return QTpsc
        
    def assembledEmbeddedMatrix(self, a, backend="PETSc"):
        self.A = fd.assemble(a).M.handle
        self.QT = self.assemble(backend)
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
            
         
