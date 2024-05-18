'''
This module test that the environment is correctly been setup.
In particular it will test for: petsc4py, PETSc, NGSolve, Netgen and ngsPETSc
'''

import petsc4py
from petsc4py import PETSc

import ngsolve as ngs
from netgen.geom2d import unit_square
from ngsolve import x,y

import ngsPETSc

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

def test_ngs():
    '''
    Testing that NGSolve can be imported correctly
    '''
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.1))
    coefficientFunction = x*(1-y)
    mip = mesh(0.2, 0.2)
    coefficientFunction(mip)
    print(mip)

def test_ngsPETSc():
    '''
    Testing that ngsPETSc can be imported correctly
    '''
    assert ngsPETSc.VERSION == "0.0.5"
