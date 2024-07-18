from ngsolve import *
import netgen.gui
from netgen.occ import *
import netgen.meshing as ngm
from mpi4py.MPI import COMM_WORLD

if COMM_WORLD.rank == 0:
    d = 0.01
    box = Box ( (-d/2,-d/2,0), (d/2,d/2,0.1) ) + Box( (-d/2, -3*d/2,0.1), (d/2, 3*d/2, 0.1+d) )
    box.faces.Min(Z).name = "bottom"
    box.faces.Max(Z).name = "top"
    mesh = Mesh(OCCGeometry(box).GenerateMesh(maxh=0.003).Distribute(COMM_WORLD))
else:
    mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))

E, nu = 210, 0.2
mu  = E / 2 / (1+nu)
lam = E * nu / ((1+nu)*(1-2*nu))

def C(u):
    F = Id(u.dim) + Grad(u)
    return F.trans * F

def NeoHooke (C):
    return 0.5*mu*(Trace(C-Id(3)) + 2*mu/lam*Det(C)**(-lam/2/mu)-1)

loadfactor = Parameter(1)
force = loadfactor * CF ( (-y, x, 0) )

fes = H1(mesh, order=2, dirichlet="bottom", dim=mesh.dim)
u,v = fes.TnT()

a = BilinearForm(fes, symmetric=True)
a += Variation(NeoHooke(C(u)).Compile()*dx)
a += ((Id(3)+Grad(u.Trace()))*force)*v*ds("top")
from ngsPETSc import NonLinearSolver
gfu_petsc = GridFunction(fes)
numsteps = 30
for step in range(numsteps):
    print("step", step)
    loadfactor.Set(720*step/numsteps)
    solver = NonLinearSolver(fes, a=a, objective=False,
                             solverParameters={"snes_type": "newtonls",
                                               "snes_max_it": 200,
                                               "snes_monitor": "",
                                               "ksp_type": "preonly",
                                               "pc_type": "lu",
                                               "snes_linesearch_type": "basic",
                                               "snes_linesearch_damping": 1.0})
    gfu_petsc = solver.solve(gfu_petsc)
    E = NeoHooke(C(gfu_petsc))
    C_=C(gfu_petsc).MakeVariable()
    PK2 = NeoHooke(C_).Diff(C_)
    F = Id(3) + Grad(gfu_petsc)
    PK1 = F * PK2
    sigma = 1/Det(F) * PK1 * F.trans
    VM = 0.5 * ((sigma[0,0]-sigma[1,1])**2+(sigma[1,1]-sigma[2,2])**2 \
                +(sigma[2,2]-sigma[0,0])**2+6*(sigma[0,1]**2+sigma[1,2]**2 \
                +sigma[2,0]**2))
    VM = sqrt(VM)
    vtk = VTKOutput(ma=mesh, coefs=[gfu_petsc, E, PK2, VM],
                names = ["u", "E", "PK2", "VM"],
                filename=f"output/hyperelasticity_{step}",
                subdivision=2)
    vtk.Do()
