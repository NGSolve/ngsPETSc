'''
This module contains all the functions related to wrapping NGSolve meshes to
PETSc DMPlex using the petsc4py interface.
'''
import itertools
import numpy as np

from petsc4py import PETSc

import netgen.meshing as ngm
try:
    import ngsolve as ngs
except ImportError:
    class ngs:
        "dummy class"
        class comp:
            "dummy class"
            Mesh = type(None)

from mpi4py import MPI

FACE_SETS_LABEL = "Face Sets"
CELL_SETS_LABEL = "Cell Sets"
EDGE_SETS_LABEL = "Edge Sets"

class MeshMapping:
    '''
    This class creates a mapping between Netgen/NGSolve meshes and PETSc DMPlex  

    :arg mesh: the mesh object, it can be either a Netgen/NGSolve mesh or a PETSc DMPlex

    :arg name: the name of to be assigned to the PETSc DMPlex, by default this is set to "Default"

    '''

    def __init__(self, mesh=None, comm=MPI.COMM_WORLD, geo=None, name="Default"):
        self.name = name
        self.comm = comm
        self.geo = geo
        if isinstance(mesh,(ngs.comp.Mesh,ngm.Mesh)):
            self.createPETScDMPlex(mesh)
        elif isinstance(mesh,PETSc.DMPlex):
            self.createNGSMesh(mesh)
        else:
            raise ValueError("Mesh format not recognised.")

    def createNGSMesh(self, plex):
        '''
        This function generate an NGSolve mesh from a PETSc DMPlex 

        :arg plex: the PETSc DMPlex to be converted in NGSolve mesh object

        '''
        self.petscPlex = plex
        ngMesh = ngm.Mesh(dim=plex.getCoordinateDim())
        self.ngMesh = ngMesh
        if self.geo:
            self.ngMesh.SetGeometry(self.geo)
        if plex.getDimension() == 2:
            coordinates = plex.getCoordinates().getArray().reshape([-1,2])
            self.ngMesh.AddPoints(coordinates)
            cStart,cEnd = plex.getHeightStratum(0)
            vStart, _ = plex.getHeightStratum(2)
            cells = []
            for i in range(cStart,cEnd):
                sIndex  = plex.getCone(i)
                if len(sIndex)==3:
                    S = list(itertools.chain.from_iterable([
                        list(plex.getCone(sIndex[k])-vStart) for k in range(len(sIndex))]))
                    cell = list(dict.fromkeys(S))
                    A = np.array([coordinates[cell[1]]-coordinates[cell[0]],
                                 coordinates[cell[2]]-coordinates[cell[1]]])
                    if np.linalg.det(A) > 0:
                        cells = cells + [cell]
                    else:
                        cells = cells + [[cell[0],cell[2],cell[1]]]
                else:
                    S = [plex.getCone(sIndex[k])-vStart for k in list(range(len(sIndex)))]
                    k = 0
                    cell = []
                    while k <  len(sIndex):
                        if S[k][0] == S[k-1][1]:
                            cell = cell+[S[k][0]]
                        else:
                            if k == 0  and S[k][0] not in list(S[k+1]):
                                S[-1] = np.array([S[-1][1],S[-1][0]])
                            else:
                                S[k] = np.array([S[k][1],S[k][0]])
                            cell = cell+[S[k][0]]
                        k = k+1
                    cells = cells + [cell]
            self.ngMesh.Add(ngm.FaceDescriptor(bc=1))
            self.ngMesh.AddElements(dim=2, index=1, data=np.asarray(cells, dtype=np.int32), base=0)
            for bcLabel in range(1,plex.getLabelSize(FACE_SETS_LABEL)+1):
                bcIndices = plex.getStratumIS("Face Sets",bcLabel).indices
                for j in bcIndices:
                    bcIndex = plex.getCone(j)-vStart
                    if len(bcIndex) == 2:
                        edge = ngm.Element1D([v+1 for v in bcIndex],
                                             index=bcLabel,edgenr=bcLabel-1)
                        if self.geo:
                            self.ngMesh.Add(edge, project_geominfo=True)
                        else:
                            self.ngMesh.Add(edge)
        elif plex.getDimension() == 3:
            coordinates = plex.getCoordinates().getArray().reshape([-1,3])
            self.ngMesh.AddPoints(coordinates)
            cStart, cEnd = plex.getHeightStratum(0)
            vStart, _ = plex.getHeightStratum(3)
            cells = []
            for i in range(cStart,cEnd):
                fIndex  = plex.getCone(i)
                f1 = plex.getCone(fIndex[0])
                f2 = plex.getCone(fIndex[1])
                sIndex = [f1[0],f1[1],f1[2],f2[0],f2[1],f2[2]]
                if len(fIndex) == 4:
                    S = list(itertools.chain.from_iterable([
                        list(plex.getCone(sIndex[k])-vStart) for k in list(range(len(sIndex)))]))
                    cell = list(dict.fromkeys(S))
                    A = np.array([coordinates[cell[1]]-coordinates[cell[0]],
                                  coordinates[cell[2]]-coordinates[cell[1]],
                                  coordinates[cell[3]]-coordinates[cell[2]]])
                    if np.linalg.det(A) > 0:
                        cells = cells + [cell]
                    else:
                        cells = cells + [[cell[0],cell[1],cell[3],cell[2]]]
            #fd = self.ngmesh.Add(ngm.FaceDescriptor(bc=plex.getLabelSize(FACE_SETS_LABEL)+1))
            self.ngMesh.Add(ngm.FaceDescriptor(bc=plex.getLabelSize(FACE_SETS_LABEL)+1))
            self.ngMesh.AddElements(dim=3, index=plex.getLabelSize(FACE_SETS_LABEL)+1,
                                    data=np.asarray(cells,dtype=np.int32), base=0)
            for bcLabel in range(1,plex.getLabelSize(FACE_SETS_LABEL)+1):
                faces = []
                bcIndices = plex.getStratumIS("Face Sets",bcLabel).indices
                for j in bcIndices:
                    sIndex  = plex.getCone(j)
                    if len(sIndex)==3:
                        S = list(itertools.chain.from_iterable([
                            list(plex.getCone(sIndex[k])-vStart) for k in range(len(sIndex))]))
                        face = list(dict.fromkeys(S))
                        A = np.array([coordinates[face[1]]-coordinates[face[0]],
                                      coordinates[face[2]]-coordinates[face[1]],
                                      coordinates[face[0]]-coordinates[face[2]]])
                        eig = np.linalg.eig(A)[0].real
                        if eig[1]*eig[2] > 0:
                            faces = faces + [face]
                        else:
                            faces = faces + [[face[0],face[2],face[1]]]
                self.ngMesh.Add(ngm.FaceDescriptor(bc=bcLabel, surfnr=bcLabel))
                if self.geo:
                    self.ngMesh.AddElements(dim=2, index=bcLabel,
                        data=np.asarray(faces,dtype=np.int32), base=0, project_geometry=True)
                else:
                    self.ngMesh.AddElements(dim=2, index=bcLabel,
                                        data=np.asarray(faces,dtype=np.int32), base=0)
        else:
            raise NotImplementedError("No implementation for dimension greater than 3.")

    def createPETScDMPlex(self, mesh):
        '''
        This function generate an PETSc DMPlex from a Netgen/NGSolve mesh object

        :arg plex: the Netgen/NGSolve mesh object to be converted into a PETSc
                   DMPlex.

        '''
        if isinstance(mesh,ngs.comp.Mesh):
            self.ngMesh = mesh.ngmesh
        else:
            self.ngMesh = mesh
        comm = self.comm
        self.geo = self.ngMesh.GetGeometry()
        if self.ngMesh.dim == 3:
            if comm.rank == 0:
                V = self.ngMesh.Coordinates()
                T = self.ngMesh.Elements3D().NumPy()["nodes"]
                T = np.array([list(np.trim_zeros(a, 'b')) for a in list(T)])-1
                surfMesh, dim = False, 3
                if len(T) == 0:
                    surfMesh, dim = True, 2
                    T = self.ngMesh.Elements2D().NumPy()["nodes"]
                    T = np.array([list(np.trim_zeros(a, 'b')) for a in list(T)])-1
                plex = PETSc.DMPlex().createFromCellList(dim, T, V, comm=comm)
                plex.setName(self.name)
                vStart, _ = plex.getDepthStratum(0)
                nMat = len(self.ngMesh.GetRegionNames(dim=3))
                nBnd = len(self.ngMesh.GetRegionNames(dim=2))
                nBBnd = len(self.ngMesh.GetRegionNames(dim=1))
                if surfMesh:
                    for e in self.ngMesh.Elements1D():
                        join = plex.getJoin([vStart+v.nr-1 for v in e.vertices])
                        plex.setLabelValue(FACE_SETS_LABEL, join[0], int(e.surfaces[1]))
                else:
                    if nMat > 1:
                        for e in self.ngMesh.Elements3D():
                            join = plex.getFullJoin([vStart+v.nr-1 for v in e.vertices])
                            plex.setLabelValue(CELL_SETS_LABEL, join[0], int(e.index))
                            cone  = plex.getCone(join[0])
                            for i in range(len(cone)):
                                plex.setLabelValue(FACE_SETS_LABEL, cone[i], nBBnd+nBnd+int(e.index))
                    for e in self.ngMesh.Elements2D():
                        join = plex.getFullJoin([vStart+v.nr-1 for v in e.vertices])
                        plex.setLabelValue(FACE_SETS_LABEL, join[0], int(e.index))
                    for e in self.ngMesh.Elements1D():
                        join = plex.getJoin([vStart+v.nr-1 for v in e.vertices])
                        plex.setLabelValue(EDGE_SETS_LABEL, join[0], int(e.index))
                self.petscPlex = plex
            else:
                plex = PETSc.DMPlex().createFromCellList(3,
                                                        np.zeros((0, 4), dtype=np.int32),
                                                        np.zeros((0, 3), dtype=np.double),
                                                        comm=comm)
                self.petscPlex = plex
        elif self.ngMesh.dim == 2:
            if comm.rank == 0:
                V = self.ngMesh.Coordinates()
                T = self.ngMesh.Elements2D().NumPy()["nodes"]
                T = np.array([list(np.trim_zeros(a, 'b')) for a in list(T)])-1
                plex = PETSc.DMPlex().createFromCellList(2, T, V, comm=comm)
                plex.setName(self.name)
                vStart, _ = plex.getDepthStratum(0)   # vertices
                nMat = len(self.ngMesh.GetRegionNames(dim=2))
                nBnd = len(self.ngMesh.GetRegionNames(dim=1))
                if nMat > 1:
                    for e in self.ngMesh.Elements2D():
                        join = plex.getFullJoin([vStart+v.nr-1 for v in e.vertices])
                        plex.setLabelValue(CELL_SETS_LABEL, join[0], int(e.index))
                        cone  = plex.getCone(join[0])
                        for i in range(len(cone)):
                            plex.setLabelValue(FACE_SETS_LABEL, cone[i], nBnd+int(e.index))
                for e in self.ngMesh.Elements1D():
                    join = plex.getJoin([vStart+v.nr-1 for v in e.vertices])
                    plex.setLabelValue(FACE_SETS_LABEL, join[0], int(e.index))
                self.petscPlex = plex
            else:
                plex = PETSc.DMPlex().createFromCellList(2,
                                                        np.zeros((0, 3), dtype=np.int32),
                                                        np.zeros((0, 2), dtype=np.double),
                                                        comm=comm)
                self.petscPlex = plex
