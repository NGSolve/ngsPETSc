'''
Pytest configuration for the ngsPETSc test suite
'''
import importlib.util

import pytest


def pytest_collection_modifyitems(items):
    '''Skip tests marked ngsolve_skip when NGSolve is not installed.'''
    if importlib.util.find_spec("ngsolve") is not None:
        return
    skip = pytest.mark.skip(reason="NGSolve is not installed")
    for item in items:
        if "ngsolve_skip" in item.keywords:
            item.add_marker(skip)
