'''
This file contains all the functions related to creating a 
Perfectly Matched Layer (PML) in Firedrake.
'''
from collections.abc import Iterable

try:
    import firedrake as fd
except ImportError:
    fd = None

import numpy as np

class WeightedMeasure(fd.ufl.Measure):
    """
    This class creates a weighted measure in Firedrake.
    """
    def __init__(self, *args, weight=None, **kwargs):
        self.weight = weight
        fd.ufl.Measure.__init__(self, *args, **kwargs)

    def __rmul__(self, integrand):
        return fd.ufl.Measure.__rmul__(self, self.weight * integrand)

class PML:
    """
    This class creates a Perfectly Matched Layer (PML) in Firedrake.
    :arg mesh: The Firedrake mesh.
    :arg order: The function space order to solve the Poisson problem, when 
                constructing the PML.
    :arg pmls: A list of touple of the from:
                (pml_region_name, pml_inner_boundary_name, pml_outer_boundary_name).
    :arg alpha: The attenuation coefficient of the PML, it can be a Constant, a Function,
                or a list, if it is a list we assume each element of the list the attenuation
                coefficient for one of the PML regions.
    :arg solver_parameters: The solver parameters for the PML problem.
    """
    def __init__(self, mesh, order, pml_regions, alpha=fd.Constant(1j), solver_parameters=None):
        """
        This function initializes the PML object.
        """
        self.mesh = mesh
        self.order = order
        self.pml_regions = pml_regions
        self.alpha = alpha
        if solver_parameters is None:
            solver_parameters = {"ksp_type":"preonly", "pc_type":"lu",
                                 "pc_factor_mat_solver_type":"mumps"}
        if not hasattr(self.mesh, "netgen_mesh"):
            raise ValueError("The mesh must be a Netgen mesh.")
        labels1 = dict(map(lambda i,j : (i,j+1) , mesh.netgen_mesh.GetRegionNames(dim=1),
                        range(len(mesh.netgen_mesh.GetRegionNames(dim=1)))))
        labels2 = dict(map(lambda i,j : (i,j+1) , mesh.netgen_mesh.GetRegionNames(dim=2),
                        range(len(mesh.netgen_mesh.GetRegionNames(dim=2)))))
        if not isinstance(alpha, (fd.Constant, fd.Function, list)):
            raise ValueError("The attenuation coefficient must be a \
                             Constant, a Function, or a list.")
        if not isinstance(alpha, Iterable):
            alpha = [alpha]*len(pml_regions)
        #Construct the PML for each PML region
        self.dalets = []
        self.V = fd.FunctionSpace(self.mesh, "CG", self.order)
        u = fd.TrialFunction(self.V)
        v = fd.TestFunction(self.V)
        for region in self.pml_regions:
            if region[0] not in labels2:
                raise ValueError("The PML region name must be a valid region name.")
            if region[1] not in labels1:
                raise ValueError("The PML inner boundary name must be a valid region name.")
            if region[2] not in labels1:
                raise ValueError("The PML outer boundary name must be a valid region name.")
            #Construct the weight function for the PML
            F = fd.inner(fd.grad(u), fd.grad(v))*fd.dx(labels2[region[0]])
            for i in range(1, len(self.mesh.netgen_mesh.GetRegionNames(dim=2))+1):
                if i != labels2[region[0]]:
                    F += fd.inner(u, v)*fd.dx(i)
            L = fd.inner(fd.Constant(1),v)*fd.dx(labels2[region[0]])
            bcs = [fd.DirichletBC(self.V, 0, labels1[region[1]]),
                   fd.DirichletBC(self.V, 1, labels1[region[2]])]
            dalet = fd.Function(self.V)
            fd.solve(F == L, dalet, bcs=bcs, solver_parameters=solver_parameters)
            self.dalets = self.dalets + [dalet]

    def adiabatic_layer(self, k, deg_absorb=2):
        """
        This function returns a complex absorption coefficient, simulating a PML
        """
        RT = 1.0e-6
        wave_len = 2*np.pi / k  # wavelength
        d_absorb = 2 * wave_len    # depth of absorber
        sigma0 = -(deg_absorb + 1) * np.log(RT) / (2.0 * d_absorb)
        absk = fd.assemble(fd.interpolate(fd.Constant(0), self.V))
        for dalet in self.dalets:
            absk += fd.assemble(fd.interpolate(self.alpha*k*sigma0*(fd.real(dalet)**deg_absorb)
                                               -(sigma0*fd.real(dalet))**2, self.V))
        fd.File("absk.pvd").write(absk)
        return absk
