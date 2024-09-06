from firedrake import *
from firedrake import *
from petsc4py import PETSc
import numpy as np
from ngsPETSc import TrefftzEmbedding
from petsc4py import PETSc
import numpy as np
from ngsPETSc import TrefftzEmbedding


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

    ATF = E.assembledEmbeddedMatrix(aDG,backend="scipy")
    LTF = E.assembledEmbeddedLoad(L)

    # Solver linear system using ksp
    ksp = PETSc.KSP().create()
    ksp.setOperators(ATF)
    ksp.setFromOptions()
    x = ATF.createVecRight()
    ksp.solve(LTF, x)
    uTF = E.embed(x)
    uTF.rename("uTF")

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
    ATF = E.assembledEmbeddedMatrix(aDG,backend="scipy")
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