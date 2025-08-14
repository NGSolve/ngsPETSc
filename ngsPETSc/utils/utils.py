import numpy as np
from petsc4py import PETSc
from scipy.spatial.distance import cdist

__all__ = ["find_permutation"]

@PETSc.Log.EventDecorator()
def find_permutation(points_a, points_b, tol=1e-5):
    """ Find all permutations between a list of two sets of points.

    Given two numpy arrays of shape (ncells, npoints, dim) containing
    floating point coordinates for each cell, determine each index
    permutation that takes `points_a` to `points_b`. Ie:
    ```
    permutation = find_permutation(points_a, points_b)
    assert np.allclose(points_a[permutation], points_b, rtol=0, atol=tol)
    ```
    """
    if points_a.shape != points_b.shape:
        raise ValueError("`points_a` and `points_b` must have the same shape.")

    p = [np.where(cdist(a, b).T < tol)[1] for a, b in zip(points_a, points_b)]
    try:
        permutation = np.array(p, ndmin=2)
    except ValueError as e:
        raise ValueError(
            "It was not possible to find a permutation for every cell"
            " within the provided tolerance"
        ) from e

    if permutation.shape != points_a.shape[0:2]:
        raise ValueError(
            "It was not possible to find a permutation for every cell"
            " within the provided tolerance"
        )

    return permutation

