'''
This module contains all the function and class needed to wrap a PETSc Nullspace in NGSolve
'''
from petsc4py import PETSc
from ngsPETSc import VectorMapping

class NullSpace:
    '''
    This class creates a PETSc Null space from NGSolve vectors

    :arg fes: NGSolve finite element space.
    :arg span: list NGSolve grid functions spanning the null space.

    '''
    def __init__(self, fes, span, near=False):
        constant = False
        self.vecMap = VectorMapping(fes)
        self.near = near
        if isinstance(span, list):
            nullspace = list(span)
            petscNullspace = []
            for vec in nullspace:
                if isinstance(vec, str):
                    if vec == "constant":
                        constant = True
                    else:
                        raise ValueError("Invalid nullspace string")
                else:
                    petscVec  = self.vecMap.petscVec(vec)
                    petscNullspace.append(petscVec)
        elif isinstance(span, str):
            if span == "constant":
                constant = True
                petscNullspace = []
            else:
                raise ValueError("Invalid nullspace string")
        else:
            raise ValueError("Invalid nullspace type")
            # Create vector space basis and orthogonalize
        self.constant = constant
        self.orthonormalize(petscNullspace)
        self.nullspace = PETSc.NullSpace().create(constant=constant, vectors=petscNullspace)
    def orthonormalize(self, basis):
        """Orthonormalize the basis."""
        for i, vec in enumerate(basis):
            alphas = []
            for vec_ in basis[:i]:
                alphas.append(vec.dot(vec_))
            for alpha, vec_ in zip(alphas, basis[:i]):
                vec.axpy(-alpha, vec_)
            if self.constant:
                # Subtract constant mode
                alpha = vec.sum()
                vec.array -= alpha
            vec.normalize()
