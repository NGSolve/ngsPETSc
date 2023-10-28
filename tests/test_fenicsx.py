'''
This module test the utils.fenicsx class
'''
import pytest

def test_square_netgen():
    '''
    Testing FEniCSx interface with Netgen generating a square mesh
    '''
    from mpi4py import MPI
    import ngsPETSc.utils.fenicsx as fx
    from dolfinx.io import XDMFFile
    try:
        pass
    except ImportError:
        pytest.skip(msg="DOLFINx unavailable, skipping FENICSx test")

    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle((0,0),(1,1))
    geoModel = fx.GeometricModel(geo, MPI.COMM_WORLD)
    domain  =geoModel .model_to_mesh(hmax=0.1)
    with XDMFFile(domain.comm, "XDMF/mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)

if __name__ == "__main__":
    test_square_netgen()
