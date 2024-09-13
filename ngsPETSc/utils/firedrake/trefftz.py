try:
    import firedrake as fd
    from petsc4py import PETSc
    from firedrake.__future__ import interpolate
except ImportError:
    fd = None
from ngsPETSc.plex import CELL_SETS_LABEL, FACE_SETS_LABEL
class TrefftzEmbedding(object):

    def __init__(self, V, b, dim=None, tol=1e-12, backend="scipy"):
        self.V = V
        self.b = b
        self.dim = V.dim() if not dim else dim + 1
        self.tol = tol
        self.backend = backend
    
    def assemble(self):
        self.B = fd.assemble(self.b).M.handle
        if self.backend == "scipy":
            import scipy.sparse as sp
            indptr, indices, data = self.B.getValuesCSR()
            Bsp = sp.csr_matrix((data, indices, indptr), shape=self.B.getSize())
            _, sig, VT = sp.linalg.svds(Bsp, k=self.dim-1, which="SM")
            QT = sp.csr_matrix(VT[0:sum(sig<self.tol), :])
            QTpsc = PETSc.Mat().createAIJ(size=QT.shape, csr=(QT.indptr, QT.indices, QT.data))
            self.dimT = QT.shape[0]
            self.sig = sig
            return QTpsc, sig
        
    def embeddedMatrix(self, a):
        self.A = fd.assemble(a).M.handle
        self.QT, _ = self.assemble()
        self.Q = PETSc.Mat().createTranspose(self.QT)
        pscQTAQ = self.QT @ self.A @ self.Q
        return pscQTAQ
    
    def embeddedMatrixAction(self, a):
        self.A = fd.assemble(a).M.handle
        self.QT, _ = self.assemble()
        pythonQTAQ = self.embeddedMatrixWrap(self.QT, self.A)
        pscQTAQ = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        pscQTAQ.setSizes(self.dimT, self.dimT)
        pscQTAQ.setType("python")
        pscQTAQ.setPythonContext(pythonQTAQ)
        pscQTAQ.setUp()
        return pscQTAQ

    def embeddedPreconditioner(self, a):
        self.A = fd.assemble(a).M.handle
        self.QT, _ = self.assemble()
        pythonQTAQ = self.embeddedPreconditioner(self, a)
        pscQTAQ = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        pscQTAQ.setSizes(self.dim, self.dim)
        pscQTAQ.setType("python")
        pscQTAQ.setPythonContext(pythonQTAQ)
        pscQTAQ.setUp()
        return pscQTAQ

    def embeddedLoad(self, L):
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
    def embedVec(self, y):
        w = self.QT.createVecRight()
        self.QT.multTranspose(y, w)
        return w

        
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

    class embeddedPreconditioner(object):
        """
        This class wraps a PETSc Preconditioner as PETSc Python matrix
        """
        def __init__(self, E, a):
            self.E = E
            self.QTAQ = self.E.embeddedMatrix(a)
            self.ksp = PETSc.KSP().create()
            self.ksp.setOperators(self.QTAQ)
            self.ksp.getPC().setType("lu")
            self.ksp.setUp()

        def mult(self, mat, X, Y): #pylint: disable=W0613
            """
            PETSc matrix-vector product
            """
            eX = self.QTAQ.createVecLeft()
            eY = self.QTAQ.createVecRight()
            self.E.QT.mult(X,eX)
            self.ksp.solve(eX, eY)
            self.E.embedVec(eY).copy(Y)

class AggregationEmbedding(TrefftzEmbedding):
    def __init__(self, V, mesh, polyMesh, dim=None, tol=1e-12):
        # Relabel facets that are inside an aggregated region
        offset = len(mesh.netgen_mesh.GetRegionNames(dim=1))\
               + len(mesh.netgen_mesh.GetRegionNames(dim=2))
        nPoly = int(max(polyMesh.dat.data[:])) # Number of aggregates
        getIdx = mesh._cell_numbering.getOffset
        plex = mesh.topology_dm
        pStart,pEnd = plex.getDepthStratum(2)
        self.facet_index = []
        for poly in range(nPoly+1):
            facets = []
            for i in range(pStart,pEnd):
                if polyMesh.dat.data[getIdx(i)] == poly:
                    for f in plex.getCone(i):
                        if f in facets:
                            plex.setLabelValue(FACE_SETS_LABEL,f,offset+poly)
                            if offset+poly not in self.facet_index:
                                self.facet_index = self.facet_index + [offset+poly]
                    facets = facets + list(plex.getCone(i))
        self.mesh = fd.Mesh(plex)
        h = fd.CellDiameter(self.mesh)
        n = fd.FacetNormal(self.mesh)
        W = fd.FunctionSpace(self.mesh, V.ufl_element())
        u = fd.TrialFunction(W)
        v = fd.TestFunction(W)
        self.b = fd.Constant(0)*fd.inner(u,v)*fd.dx
        for i in self.facet_index:
            self.b += fd.inner(fd.jump(u),fd.jump(v))*fd.dS(i)
        super().__init__(W, self.b, dim, tol)
        

def jumpNormal(u,n):
    return 0.5*fd.dot(n, (fd.grad(u)("+")-fd.grad(u)("-")))

def dumpAggregation(mesh):
    if mesh.comm.size > 1:
        raise NotImplementedError("Parallel mesh aggregation not supported")
    plex = mesh.topology_dm
    pStart,pEnd = plex.getDepthStratum(2)
    eStart,eEnd = plex.getDepthStratum(1)
    adjacency = []
    print(pStart,pEnd)
    print(eStart,eEnd)
    for i in range(pStart,pEnd):
        ad = plex.getAdjacency(i)
        print(ad)
        local = []
        for a in ad:
            print("\t{}".format(plex.getSupport(a)))
            supp = plex.getSupport(a)
            supp = supp[supp<eEnd]
            for s in supp:
                if s < pEnd and s != ad[0]:
                    local = local + [s] 
        adjacency = adjacency + [(i, local)]
    adjacency = sorted(adjacency, key=lambda x: len(x[1]))[::-1]
    u = fd.Function(fd.FunctionSpace(mesh,"DG",0))

    getIdx = mesh._cell_numbering.getOffset
    av = list(range(pStart,pEnd))
    col = 0
    for a in adjacency:
        if a[0] in av:
            for k in a[1]:
                if k in av:
                    av.remove(k)
                    u.dat.data[getIdx(k)] = col
            av.remove(a[0])
            u.dat.data[getIdx(a[0])] = col
            col = col + 1
    print(adjacency)
    print(av)
    return u