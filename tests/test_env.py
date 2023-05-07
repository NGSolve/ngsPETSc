'''
This module test that the environment is correctly been setup.
In particular it will test for: petsc4py, PETSc, NGSolve, Netgen and ngsPETSc
'''

import petsc4py
from petsc4py import PETSc

def test_petsc4py():
    '''
    Testing that petsc4py can be imported correctly
    '''
    assert petsc4py.get_config() != ""

def  test_petsc():
    '''
    Testing that PETSc can be imported correctly
    '''
    assert PETSc.DECIDE == -1
