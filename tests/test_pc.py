'''
This module test the pc class
'''
from math import sqrt

from ngsolve import VectorH1, H1, HDiv, HCurlDiv, TangentialFacetFESpace, L2
from ngsolve import Mesh, BilinearForm, LinearForm, Integrate
from ngsolve import x, y, dx, ds, grad, InnerProduct, CF, div, specialcf
from ngsolve import Sym, Grad
from ngsolve import Preconditioner, GridFunction, ConvertOperator
from ngsolve import COUPLING_TYPE, Compress, IntRange
from ngsolve.solvers import CG
from ngsolve.krylovspace import CGSolver
from ngsolve.la import EigenValues_Preconditioner
import netgen.meshing as ngm

from mpi4py.MPI import COMM_WORLD
import pytest

from ngsPETSc import pc 

def test_pc():
    '''
    Testing the pc has registered function to register preconditioners
    '''
    assert hasattr(pc,"createPETScPreconditioner")

def test_pc_gamg():
    '''
    Testing the PETSc GAMG solver
    '''
    from netgen.geom2d import unit_square
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    fes = H1(mesh, order=4, dirichlet="left|right|top|bottom")
    u,v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx)
    a.Assemble()
    pre = Preconditioner(a, "PETScPC", pc_type="gamg")
    f = LinearForm(fes)
    f += 32 * (y*(1-y)+x*(1-x)) * v * dx
    f.Assemble()
    gfu = GridFunction(fes)
    gfu.vec.data = CG(a.mat, rhs=f.vec, pre=pre, printrates=mesh.comm.rank==0)
    exact = 16*x*(1-x)*y*(1-y)
    assert sqrt(Integrate((gfu-exact)**2, mesh))<1e-4

@pytest.mark.mpi_skip()
def test_pc_hiptmaier_xu_sor():
    '''
    Testing Hiptmaier Xu preconditioner with SOR smoother
    This test doesn't work in parallel becasue SOR is not implemented has no
    parallel implementation in PETSc. 
    '''
    from netgen.geom2d import unit_square
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    order=4
    fesDG = L2(mesh, order=order, dgjumps=True)
    u,v = fesDG.TnT()
    aDG = BilinearForm(fesDG)
    jump_u = u-u.Other()
    jump_v = v-v.Other()
    n = specialcf.normal(2)
    mean_dudn = 0.5*n * (grad(u)+grad(u.Other()))
    mean_dvdn = 0.5*n * (grad(v)+grad(v.Other()))
    alpha = 4
    h = specialcf.mesh_size
    aDG = BilinearForm(fesDG)
    aDG += grad(u)*grad(v) * dx
    aDG += alpha*order**2/h*jump_u*jump_v * dx(skeleton=True)
    aDG += alpha*order**2/h*u*v * ds(skeleton=True)
    aDG += (-mean_dudn*jump_v -mean_dvdn*jump_u) * dx(skeleton=True)
    aDG += (-n*grad(u)*v-n*grad(v)*u)* ds(skeleton=True)
    aDG.Assemble()

    fDG = LinearForm(fesDG)
    fDG += 1*v * dx
    fDG.Assemble()
    gfuDG = GridFunction(fesDG)
    fesH1 = H1(mesh, order=2, dirichlet=".*")
    u,v = fesH1.TnT()
    aH1 = BilinearForm(fesH1)
    aH1 += grad(u)*grad(v)*dx
    smoother = Preconditioner(aDG, "PETScPC", pc_type="sor", pc_sor_omega=1., pc_sor_symmetric="")
    preH1 = Preconditioner(aH1, "PETScPC", pc_type="bddc", matType="is")
    aH1.Assemble()
    transform = fesH1.ConvertL2Operator(fesDG)
    pre = transform @ preH1.mat @ transform.T + smoother.mat
    CG(mat=aDG.mat, rhs=fDG.vec, sol=gfuDG.vec, pre=pre, printrates = True, maxsteps=200)
    lam = EigenValues_Preconditioner(aDG.mat, pre)
    assert (lam.NumPy()<3.0).all()        

def test_pc_hiptmaier_xu_bjacobi():
    '''
    Testing Hiptmaier Xu preconditioner with bloack Jaocobi smoother
    '''
    from netgen.geom2d import unit_square
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    order=4
    fesDG = L2(mesh, order=order, dgjumps=True)
    u,v = fesDG.TnT()
    aDG = BilinearForm(fesDG)
    jump_u = u-u.Other()
    jump_v = v-v.Other()
    n = specialcf.normal(2)
    mean_dudn = 0.5*n * (grad(u)+grad(u.Other()))
    mean_dvdn = 0.5*n * (grad(v)+grad(v.Other()))
    alpha = 4
    h = specialcf.mesh_size
    aDG = BilinearForm(fesDG)
    aDG += grad(u)*grad(v) * dx
    aDG += alpha*order**2/h*jump_u*jump_v * dx(skeleton=True)
    aDG += alpha*order**2/h*u*v * ds(skeleton=True)
    aDG += (-mean_dudn*jump_v -mean_dvdn*jump_u) * dx(skeleton=True)
    aDG += (-n*grad(u)*v-n*grad(v)*u)* ds(skeleton=True)
    aDG.Assemble()

    fDG = LinearForm(fesDG)
    fDG += 1*v * dx
    fDG.Assemble()
    gfuDG = GridFunction(fesDG)
    fesH1 = H1(mesh, order=2, dirichlet=".*")
    u,v = fesH1.TnT()
    aH1 = BilinearForm(fesH1)
    aH1 += grad(u)*grad(v)*dx
    smoother = Preconditioner(aDG, "PETScPC", pc_type="bjacobi")
    preH1 = Preconditioner(aH1, "PETScPC", pc_type="bddc", matType="is")
    aH1.Assemble()
    transform = fesH1.ConvertL2Operator(fesDG)
    pre = transform @ preH1.mat @ transform.T + smoother.mat
    CG(mat=aDG.mat, rhs=fDG.vec, sol=gfuDG.vec, pre=pre, printrates = True, maxsteps=200)
    lam = EigenValues_Preconditioner(aDG.mat, pre)
    assert (lam.NumPy()<3.0).all()        

@pytest.mark.skip()
def test_pc_auxiliary_mcs():
    from netgen.occ import X, Rectangle, OCCGeometry

    shape = Rectangle(2,0.41).Circle(0.2,0.2,0.05).Reverse().Face()
    shape.edges.name="wall"
    shape.edges.Min(X).name="inlet"
    shape.edges.Max(X).name="outlet"

    mesh = OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.1, comm=COMM_WORLD)
    for l in range(3):
        mesh.Refine()
    mesh = Mesh(mesh)
    mesh.Curve(3)
    order=2
    nu=0.001
    inflow="inlet"
    outflow="outlet"
    wall="wall"
    V = HDiv(mesh, order=order, dirichlet=inflow+"|"+wall, RT=False)
    Vhat = TangentialFacetFESpace(mesh, order=order-1, dirichlet=inflow+"|"+wall+"|"+outflow)
    Sigma = HCurlDiv(mesh, order = order-1, orderinner=order, discontinuous=True)
    S = L2(mesh, order=order-1)            

    Sigma.SetCouplingType(IntRange(0,Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
    Sigma = Compress(Sigma)
    S.SetCouplingType(IntRange(0,S.ndof), COUPLING_TYPE.HIDDEN_DOF)
    S = Compress(S)
            
    X = V*Vhat*Sigma*S
    for i in range(X.ndof):
        if X.CouplingType(i) == COUPLING_TYPE.WIREBASKET_DOF:
            X.SetCouplingType(i, COUPLING_TYPE.INTERFACE_DOF)
            
    u, uhat, sigma, W  = X.TrialFunction()
    v, vhat, tau, R  = X.TestFunction()

    def Skew2Vec(m): return m[1,0]-m[0,1]

    dS = dx(element_boundary=True)
    n = specialcf.normal(mesh.dim)
    def tang(u): return u-(u*n)*n

    a = BilinearForm (X, eliminate_hidden = True)
    a += -0.5/nu * InnerProduct(sigma,tau) * dx + \
        (div(sigma)*v+div(tau)*u) * dx + \
        (InnerProduct(W,Skew2Vec(tau)) + InnerProduct(R,Skew2Vec(sigma))) * dx + \
        -(((sigma*n)*n) * (v*n) + ((tau*n)*n )* (u*n)) * dS + \
        (-(sigma*n)*tang(vhat) - (tau*n)*tang(uhat)) * dS

    a += 10*nu*div(u)*div(v) * dx
    a.Assemble()

    uin=CF( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
    gf0 = GridFunction(X)
    gfu0,_,_,_ = gf0.components                   
    gfu0.Set(uin, definedon=mesh.Boundaries(inflow))
    gf0.components[0].vec.data = gfu0.vec
    from ngsolve import Draw
    import netgen.gui
    gf = GridFunction(X)
    gf.vec.data = gf0.vec
    Xaux = VectorH1(mesh, order=order, dirichlet=inflow+"|"+wall)
    uaux, vaux = Xaux.TnT()
    aaux = BilinearForm(nu*InnerProduct(Sym(Grad(uaux)), Grad(vaux))*dx).Assemble()
    preaux = Preconditioner(aaux, "PETScPC", pc_type="gamg")
    convu = ConvertOperator(Xaux, X.components[0], localop=True)
    convuhat = ConvertOperator(Xaux, X.components[1], localop=True)
    embu, embuhat, _, _ = X.embeddings
    conv = embu@convu+embuhat@convuhat
    a.Assemble()
    #Hiptmaier-Xu Preconditioner
    localpre = Preconditioner(a, "PETScPC", pc_type="sor", pc_sor_omega=1., pc_sor_symmetric="")
    pre = localpre + conv @ preaux @ conv.T
    inv = CGSolver(mat=a.mat, pre=pre, printing=True, maxsteps=100)
    gf.vec.data -= inv@a.mat * gf.vec
    Draw(gf.components[0], mesh, "preauxu")
    lam = EigenValues_Preconditioner(a.mat, pre)
    print(lam)
    #GAMG Preconditioner
    pre = Preconditioner(a, "PETScPC", pc_type="gamg")
    inv = CGSolver(mat=a.mat, pre=pre, printing=True, maxsteps=100)
    gf.vec.data -= inv@a.mat * gf.vec
    Draw(gf.components[0], mesh, "preu")
    lam = EigenValues_Preconditioner(a.mat, pre)
    print(lam)
if __name__ == '__main__':
    test_pc()
    test_pc_gamg()
    test_pc_hiptmaier_xu_sor()
    test_pc_hiptmaier_xu_bjacobi()