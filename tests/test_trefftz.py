from firedrake import *
from firedrake import *
from petsc4py import PETSc
import numpy as np
from ngsPETSc import TrefftzEmbedding, AggregationEmbedding
from netgen.gui import *

def test_trefftz_laplace():
    order = 6
    mesh = UnitSquareMesh(2, 2)
    x, y = SpatialCoordinate(mesh)
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    V = FunctionSpace(mesh, "DG", order)
    u = TrialFunction(V)
    v = TestFunction(V)
    def delta(u):
        return div(grad(u))
    a = inner(delta(u), delta(v)) * dx
    alpha = 4
    mean_dudn = 0.5 * dot(grad(u("+"))+grad(u("-")),n("+"))
    mean_dvdn = 0.5 * dot(grad(v("+"))+grad(v("-")),n("+"))
    aDG = inner(grad(u),grad(v))* dx 
    aDG += inner((alpha*order**2/(h("+")+h("-")))*jump(u),jump(v))*dS
    aDG += inner(-mean_dudn,jump(v))*dS-inner(mean_dvdn,jump(u))*dS
    aDG += alpha*order**2/h*inner(u,v)*ds
    aDG += -inner(dot(n,grad(u)),v)*ds -inner(dot(n,grad(v)),u)*ds
    f = Function(V).interpolate(exp(x)*sin(y)) 
    L = alpha*order**2/h*inner(f,v)*ds - inner(dot(n,grad(v)),f)*ds
    # Solve the problem
    uDG = Function(V)
    uDG.rename("uDG")
    solve(aDG == L, uDG)

    E = TrefftzEmbedding(V, a, tol=1e-8)
    ATF = E.assembledEmbeddedMatrixFree(aDG)
    LTF = E.assembledEmbeddedLoad(L)

    # Solver linear system using ksp
    ksp = PETSc.KSP().create()
    ksp.setOperators(ATF)
    ksp.setFromOptions()
    x = ATF.createVecRight()
    ksp.solve(LTF, x)
    uTF = E.embed(x)
    uTF.rename("uTF")
    #assmeble the error
    assert(assemble(inner(uDG-uTF,uDG-uTF)*dx) < 1e-6)
    assert(E.dimT < V.dim()/2)

def test_aggregation():
    order = 2
    from netgen.occ import WorkPlane, OCCGeometry, Glue, gp_Vec
    rect = WorkPlane().Rectangle(1,1).Face()
    rect.faces.name="rect"
    rect2 = WorkPlane().Rectangle(1,1).Face().Move(gp_Vec(1,0,0))
    rect2.faces.name="rect2"
    geo = OCCGeometry(Glue([rect,rect2]), dim=2)
    ngmesh = geo.GenerateMesh(maxh=0.75)
    mesh = Mesh(ngmesh)
    x, y = SpatialCoordinate(mesh)
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    V = FunctionSpace(mesh, "DG", order)

    u = TrialFunction(V)
    v = TestFunction(V)

    def delta(u):
        return div(grad(u))

    a = inner(delta(u), delta(v)) * dx
    alpha = 4
    mean_dudn = 0.5 * dot(grad(u("+"))+grad(u("-")),n("+"))
    mean_dvdn = 0.5 * dot(grad(v("+"))+grad(v("-")),n("+"))
    aDG = inner(grad(u),grad(v))* dx 
    aDG += inner((alpha*order**2/(h("+")+h("-")))*jump(u),jump(v))*dS
    aDG += inner(-mean_dudn,jump(v))*dS-inner(mean_dvdn,jump(u))*dS
    aDG += alpha*order**2/h*inner(u,v)*ds
    aDG += -inner(dot(n,grad(u)),v)*ds -inner(dot(n,grad(v)),u)*ds

    f = Function(V).interpolate(exp(x)*sin(y)) 
    L = alpha*order**2/h*inner(f,v)*ds - inner(dot(n,grad(v)),f)*ds

    E = AggregationEmbedding(V, mesh, tol=1e-8)
    ATF = E.assembledEmbeddedMatrixFree(aDG)
    LTF = E.assembledEmbeddedLoad(L)

    # Solver linear system using ksp
    ksp = PETSc.KSP().create()
    ksp.setOperators(ATF)
    ksp.setFromOptions()
    x = ATF.createVecRight()
    ksp.solve(LTF, x)
    uTF = E.embed(x)
    uTF.rename("uTF")
    x, y = SpatialCoordinate(E.mesh)
    f = Function(E.V).interpolate(exp(x)*sin(y))
    assert(assemble(inner(f-uTF,f-uTF)*dx)<1e-3)
    assert(E.dimT == V.dim()-10)

def test_aggregation_multigrid():
    import random
    from netgen.occ import WorkPlane, OCCGeometry, Glue, gp_Vec
    order = 2
    rect = WorkPlane().Rectangle(1,1).Face()
    rect.faces.name="rect"
    rect.faces.col = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    rect2 = WorkPlane().Rectangle(1,1).Face().Move(gp_Vec(1,0,0))
    rect2.faces.name="rect2"
    rect2.faces.col = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    rect3 = WorkPlane().Rectangle(1,1).Face().Move(gp_Vec(0,1,0))
    rect3.faces.name="rect3"
    rect3.faces.col = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    rect4 = WorkPlane().Rectangle(1,1).Face().Move(gp_Vec(1,1,0))
    rect4.faces.name="rect4"
    rect4.faces.col = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    geo = OCCGeometry(Glue([rect,rect2,rect3,rect4]), dim=2)
    ngmesh = geo.GenerateMesh(maxh=0.75)
    mesh = Mesh(ngmesh)
    x, y = SpatialCoordinate(mesh)
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    V = FunctionSpace(mesh, "DG", order)

    u = TrialFunction(V)
    v = TestFunction(V)

    def delta(u):
        return div(grad(u))

    a = inner(delta(u), delta(v)) * dx
    alpha = 4
    mean_dudn = 0.5 * dot(grad(u("+"))+grad(u("-")),n("+"))
    mean_dvdn = 0.5 * dot(grad(v("+"))+grad(v("-")),n("+"))
    aDG = inner(grad(u),grad(v))* dx 
    aDG += inner((alpha*order**2/(h("+")+h("-")))*jump(u),jump(v))*dS
    aDG += inner(-mean_dudn,jump(v))*dS-inner(mean_dvdn,jump(u))*dS
    aDG += alpha*order**2/h*inner(u,v)*ds
    aDG += -inner(dot(n,grad(u)),v)*ds -inner(dot(n,grad(v)),u)*ds

    f = Function(V).interpolate(exp(x)*sin(y)) 
    L = alpha*order**2/h*inner(f,v)*ds - inner(dot(n,grad(v)),f)*ds

    # Solve the problem
    uDG = Function(V)
    uDG.rename("uDG")
    solve(aDG == L, uDG, solver_parameters={"ksp_type":"cg", "pc_type": "none", "ksp_monitor":None})

    E = AggregationEmbedding(V, mesh, tol=1e-6)
    ATF = E.assembledEmbeddedMatrix(aDG)
    LTF = E.assembledEmbeddedLoad(L)

    # Solver linear system using ksp
    ksp = PETSc.KSP().create()
    ksp.setOperators(ATF)
    ksp.getPC().setType("none")
    ksp.setFromOptions()
    x = ATF.createVecRight()
    ksp.solve(LTF, x)
    uTF = E.embed(x)
    uTF.rename("uTF")
    x, y = SpatialCoordinate(E.mesh)
    f = Function(E.V).interpolate(exp(x)*sin(y))

    #Use the assembled matrix to solve the system
    A = assemble(aDG).M.handle
    PTF = E.assembledEmbeddedPreconditioner(aDG)
    ksp = PETSc.KSP().create()
    ksp.setOperators(A, PTF)
    ksp.setFromOptions()
    ksp.getPC().setType("mat")
    x = A.createVecRight()
    y = assemble(L).dat.vec
    with y as yvec:
        ksp.solve(yvec, x)
    assert(E.dim>E.dimT)
    assert(ksp.getIterationNumber() < 4)