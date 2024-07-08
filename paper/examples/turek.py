from firedrake import *
from irksome import Dt, MeshConstant, TimeStepper
import netgen 
from netgen.occ import *
from solvers import paramsLU, paramsALMG, eps

nu = Constant(1e-3); Re = 0.1/nu; gamma = Constant(1e3)
refs = 2
maxh = 0.1
degree = 4
Tf = 6.
N = int(Tf/((Tf*maxh)/(2**(refs+2))))

ksp_max_its = 50
lin_atol = 1 / (N**2)
lin_rtol = 1e-8
print(GREEN % ("Using absolute tolerance: {} and relative tolerance: {}".format(float(lin_atol), float(lin_rtol))))

params0 = paramsLU
params0["snes_monitor"] = None
params0["snes_rtol"] = lin_rtol
params0["snes_atol"] = lin_atol 
params0["ksp_monitor"] = None
params0["ksp_atol"] = lin_atol
params0["ksp_rtol"] = lin_rtol

params = paramsALMG
params["snes_monitor"] = None
params["snes_rtol"] = lin_rtol
params["snes_atol"] = lin_atol 
params["ksp_monitor"] = None
params["ksp_atol"] = lin_atol
params["ksp_rtol"] = lin_rtol

appctx = {"nu": nu, "Re": Re, "gamma": gamma}


circle = Circle(Pnt(0.2,0.2),0.05).Face()
shape = (Rectangle(2, 0.41).Face()-circle)
shape.edges.name="wall"
shape.edges.Min(X).name="inlet"
shape.edges.Max(X).name="outlet"
geo = OCCGeometry(shape, dim=2)
ngmesh = geo.GenerateMesh(maxh=maxh)


mesh = Mesh(ngmesh)
nh = MeshHierarchy(mesh, refs, netgen_flags={"degree": 4})
mesh = nh[-1]

V = VectorFunctionSpace(mesh, "CG", degree)
W = FunctionSpace(mesh, "CG", degree-1)
Z = V * W

sol = Function(Z)
u, p = TrialFunctions(Z)
v, q = TestFunctions(Z)


x, y = SpatialCoordinate(mesh)

inflowoutflow = Function(V).interpolate(as_vector([1.5*4*y*(0.41-y)/(0.41*0.41), 0]))
labelsInlet = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name in ["inlet"]]
labelsWall = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name in ["wall"]]
bcs = [DirichletBC(Z.sub(0), inflowoutflow, labelsInlet),
       DirichletBC(Z.sub(0), zero(2), labelsWall)]
nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

print(GREEN % ("Solving for initial guess with nu: {}".format(float(nu))))
#Solving Stokes for initial data

a = (nu*inner(eps(u), eps(v)) - p * div(v) + div(u) * q)*dx
L = inner(Constant((0, 0)), v) * dx
sol0 = Function(Z)
solve(a == L, sol0, bcs=bcs, solver_parameters=params0)
u0, p0 = sol0.split()
u0.rename("velocity")
p0.rename("pressure")

sol = project(sol0, Z, bcs=bcs)
#Setting up solution of Navier-Stokes
MC = MeshConstant(mesh)
dt = MC.Constant(Tf/N)
print(GREEN % ("Timestep: {}".format(float(dt))))
t = MC.Constant(0.0)

u, p = split(sol)


F = (inner(Dt(u), v) 
     + nu*inner(grad(u), grad(v))
     + inner(dot(grad(u), u), v)
     - inner(p, div(v))
     + inner(div(u),q)
     + gamma*inner(div(u),div(v))
     )*dx

from irksome import GaussLegendre
butcher_tableau = GaussLegendre(1)
ns = butcher_tableau.num_stages

stepper = TimeStepper(F, butcher_tableau, t, dt, sol, bcs=bcs,
                      solver_parameters=params, appctx=appctx)

outfile1 = File("output/NSVelocity.pvd")
outfile2 = File("output/NSPressure.pvd")
outfile1.write(sol.subfunctions[0], time=0)
outfile2.write(sol.subfunctions[1], time=0)

for j in range(N):
    print(GREEN % ("Computing solution at time: {}".format(float(t))))
    stepper.advance()
    t.assign(float(t) + float(dt))
    
    outfile1.write(sol.subfunctions[0], time=j * float(dt))
    outfile2.write(sol.subfunctions[1], time=j * float(dt))

