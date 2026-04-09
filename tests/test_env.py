'''
This module test that the environment has been setup correctly.
In particular it will test for: petsc4py, PETSc, NGSolve and Netgen
'''

def test_petsc4py():
    '''
    Testing that petsc4py can be imported correctly
    '''
    import petsc4py
    assert petsc4py.get_config() != ""


def  test_petsc():
    '''
    Testing that PETSc can be imported correctly
    '''
    from petsc4py import PETSc
    assert PETSc.DECIDE == -1


def test_netgen():
    '''
    Testing that NGSolve can be imported correctly
    '''
    from netgen.geom2d import unit_square
    unit_square.GenerateMesh(maxh=0.1)


def test_ngs():
    '''
    Testing that NGSolve can be imported correctly
    '''
    from netgen.geom2d import unit_square
    import ngsolve as ngs
    from ngsolve import x,y
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.1))
    coefficientFunction = x*(1-y)
    mip = mesh(0.2, 0.2)
    coefficientFunction(mip)
    print(mip)
