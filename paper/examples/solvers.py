from firedrake import *
from firedrake.preconditioners.base import PCBase

paramsLU = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "ksp_monitor": None,
    "ksp_converged_reason": None,
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
paramsLaplaceMG = {
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "mg",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "__main__.Mass",
    "fieldsplit_1_aux_pc_type": "bjacobi",
    "fieldsplit_1_aux_sub_pc_type": "icc",
}
paramsFullMG = {
    "ksp_type": "gcr",
    "mat_type": "nest",
    "pc_type": "mg",
    "mg_coarse_ksp_type": "preonly",
    "mg_coarse_pc_type": "fieldsplit",
    "mg_coarse_pc_fieldsplit_type": "schur",
    "mg_coarse_pc_fieldsplit_schur_fact_type": "full",
    "mg_coarse_fieldsplit_0_ksp_type": "preonly",
    "mg_coarse_fieldsplit_0_pc_type": "lu",
    "mg_coarse_fieldsplit_1_ksp_type": "preonly",
    "mg_coarse_fieldsplit_1_pc_type": "python",
    "mg_coarse_fieldsplit_1_pc_python_type": "__main__.Mass",
    "mg_coarse_fieldsplit_1_aux_pc_type": "cholesky",
    "mg_levels_ksp_type": "richardson",
    "mg_levels_ksp_max_it": 1,
    "mg_levels_pc_type": "fieldsplit",
    "mg_levels_pc_fieldsplit_type": "schur",
    "mg_levels_pc_fieldsplit_schur_fact_type": "full",
    "mg_levels_fieldsplit_0_ksp_type": "richardson",
    "mg_levels_fieldsplit_0_ksp_convergence_test": "skip",
    "mg_levels_fieldsplit_0_ksp_max_it": 2,
    "mg_levels_fieldsplit_0_pc_type": "bjacobi",
    "mg_levels_fieldsplit_1_ksp_type": "richardson",
    "mg_levels_fieldsplit_1_ksp_convergence_test": "skip",
    "mg_levels_fieldsplit_1_ksp_max_it": 3,
    "mg_levels_fieldsplit_1_pc_type": "python",
    "mg_levels_fieldsplit_1_pc_python_type": "__main__.Mass",
    "mg_levels_fieldsplit_1_aux_pc_type": "bjacobi"
}
paramsPCDLU = {"mat_type": "matfree",
             "ksp_type": "fgmres",
             "ksp_gmres_modifiedgramschmidt": None,
             "ksp_monitor_true_residual": None,
             "pc_type": "fieldsplit",
             "pc_fieldsplit_type": "schur",
             "pc_fieldsplit_schur_fact_type": "lower",
             "fieldsplit_0_ksp_type": "preonly",
             "fieldsplit_0_pc_type": "python",
             "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
             "fieldsplit_0_assembled_pc_type": "lu",
             "fieldsplit_1_ksp_type": "gmres",
             "fieldsplit_1_ksp_rtol": 1e-4,
             "fieldsplit_1_pc_type": "python",
             "fieldsplit_1_pc_python_type": "firedrake.PCDPC",
             "fieldsplit_1_pcd_Mp_ksp_type": "preonly",
             "fieldsplit_1_pcd_Mp_pc_type": "lu",
             "fieldsplit_1_pcd_Kp_ksp_type": "preonly",
             "fieldsplit_1_pcd_Kp_pc_type": "lu",
             "fieldsplit_1_pcd_Fp_mat_type": "matfree"
}

class DGMassInv(PCBase):
    def initialize(self, pc):
        _, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        V = dmhooks.get_function_space(pc.getDM())
        # get function spaces
        u = TrialFunction(V)
        v = TestFunction(V)
        massinv = assemble(Tensor(inner(u, v)*dx).inv)
        self.massinv = massinv.petscmat
        self.nu = appctx["nu"]
        self.gamma = appctx["gamma"]

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        self.massinv.mult(x, y)
        scaling = float(self.nu) + float(self.gamma)
        y.scale(-scaling)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError("Sorry!")

class Mass(AuxiliaryOperatorPC): 
    def form(self, pc, test, trial):
        appctx = self.get_appctx(pc)
        a=-1/(appctx["gamma"]+appctx["nu"])*inner(test, trial)*dx
        bcs = None
        return (a, bcs)

paramsALLU = {
    "snes_monitor": None,
    "snes_error_on_nonconvergence": False,
    'mat_type': 'nest',
    'ksp_type': 'fgmres',
    'ksp_converged_reason': None,
    'ksp_max_it': 500,
    'ksp_monitor_true_residual': None,
    'pc_fieldsplit_schur_factorization_type': 'full',
    'pc_fieldsplit_schur_precondition': 'user',
    'pc_fieldsplit_type': 'schur',
    'pc_type': 'fieldsplit',
    'fieldsplit_0': {'ksp_max_it': 1,
                        'ksp_type': 'preonly',
                        'mat_mumps_icntl_14': 150,
                        'pc_factor_mat_solver_type': 'mumps',
                        'pc_type': 'lu'},
        'fieldsplit_1': {'ksp_type': 'preonly',
                        'pc_python_type': __name__ + '.Mass',
                        'pc_type': 'python'},

}
paramsALMG = {
    'fieldsplit_0': {'ksp_convergence_test': 'skip',
                        'ksp_max_it': 1,
                        'ksp_norm_type': 'unpreconditioned',
                        'ksp_richardson_self_scale': False,
                        'ksp_type': 'richardson',
                        'mg_coarse_assembled': {'mat_type': 'aij',
                                                'pc_telescope_reduction_factor': 1,
                                                'pc_telescope_subcomm_type': 'contiguous',
                                                'pc_type': 'telescope',
                                                'telescope_pc_factor_mat_solver_type': 'superlu_dist',
                                                'telescope_pc_type': 'lu'},
                        'mg_coarse_pc_python_type': 'firedrake.AssembledPC',
                        'mg_coarse_pc_type': 'python',
                        'mg_levels': {'ksp_convergence_test': 'skip',
                                    'ksp_max_it': 10,
                                    'ksp_norm_type': 'unpreconditioned',
                                    'ksp_type': 'fgmres',
                                    'pc_python_type': 'firedrake.ASMStarPC',
                                    'pc_type': 'python'},
                        'pc_mg_log': None,
                        'pc_mg_type': 'full',
                        'pc_type': 'mg'},

    'fieldsplit_1': {'ksp_type': 'preonly',
                        'pc_python_type': __name__+'.Mass',
                        'pc_type': 'python'},
    'ksp_converged_reason': None,
    'ksp_max_it': 500,
    'ksp_monitor': None,
    'ksp_type': 'fgmres',
    'mat_type': 'nest',
    'pc_fieldsplit_schur_factorization_type': 'full',
    'pc_fieldsplit_schur_precondition': 'user',
    'pc_fieldsplit_type': 'schur',
    'pc_type': 'fieldsplit',
    'snes_converged_reason': None,
    'snes_linesearch_maxstep': 1.0,
    'snes_linesearch_monitor': None,
    'snes_linesearch_type': 'basic',
    'snes_max_it': 20,
    'snes_monitor': None,
    'snes_type': 'newtonls'
}


def eps(u):
    return 0.5*(grad(u)+grad(u).T)