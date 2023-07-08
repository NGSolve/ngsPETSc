PETSc DMPlex
------------

In this tutorial we will have an in-depth look in to the way we
transform NGSolve/Netgen ``Mesh`` to a PETSc DMPlex. In particular we
will show to create a PETSc ``DMPlex`` from an NGSolve/Netgen and
vice-versa, using the ``MeshMapping`` class. We will also show how to
use PETSc ``DMPlexTransform`` to construct purely quad mesh and Alfeld
split mesh.

Let’s test if the cluster has by initialized correctly by checking the
size of the ``COMM_WORLD``, unfortunately the ``MeshMapping`` class only
works in serial at the moment, but we are looking into how to make it
work in parallel.

.. code:: ipython3

    from ipyparallel import Cluster
    c = await Cluster().start_and_connect(n=1, activate=True)


.. parsed-literal::

    Starting 1 engines with <class 'ipyparallel.cluster.launcher.LocalEngineSetLauncher'>



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?engine/s]


.. code:: ipython3

    %%px
    from mpi4py.MPI import COMM_WORLD
    COMM_WORLD.Get_size()



.. parsed-literal::

    [0;31mOut[0:1]: [0m1


First we need to construct the distributed mesh that will be interested
in dealing with in our map to a PETSc ``DMPlex``.

.. code:: ipython3

    %%px
    from ngsolve import Mesh
    from netgen.geom2d import unit_square
    
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.2).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))

We can now convert this matrix in a PETSc ``DMPlex``, to do this we will
initialize a ``MeshMapping`` class and then access the mapped PETSc
``DMPlex`` as the attribute ``petscPlex``.

.. code:: ipython3

    %%px
    from ngsPETSc import MeshMapping
    Map = MeshMapping(mesh)
    Map.petscPlex.view()



.. parsed-literal::

    [stdout:0] DM Object: Default 1 MPI process
      type: plex
    Default in 2 dimensions:
      Number of 0-cells per rank: 37
      Number of 1-cells per rank: 88
      Number of 2-cells per rank: 52
    Labels:
      celltype: 3 strata with value/size (0 (37), 3 (52), 1 (88))
      depth: 3 strata with value/size (0 (37), 1 (88), 2 (52))
      Face Sets: 4 strata with value/size (1 (5), 2 (5), 3 (5), 4 (5))



We can use any PETSc ``DMPlex`` function that is wrapped in petsc4py on
the ``Map.petscPlex`` object. For example, we can apply any PETSc
``DMPlexTransform``. We will now apply different PETSc
``DMPlexTransform`` and check using the ``view`` method that we have the
mesh was transformed correctly. We begin with the PETSc
``DMPlexTransformType.REFINEREGULAR`` which will create split a triangle
in four subs triangles connecting the middle points of each vertex of
the triangle to create a new triangle in the center.

.. code:: ipython3

    %%px
    from petsc4py import PETSc
    tr = PETSc.DMPlexTransform().create(comm=PETSc.COMM_WORLD)
    tr.setType(PETSc.DMPlexTransformType.REFINEREGULAR)
    tr.setDM(Map.petscPlex)
    tr.setUp()
    newplex = tr.apply(Map.petscPlex)
    newplex.view()



.. parsed-literal::

    [stdout:0] DM Object: 1 MPI process
      type: plex
    DM_0x55916d78e370_1 in 2 dimensions:
      Number of 0-cells per rank: 125
      Number of 1-cells per rank: 332
      Number of 2-cells per rank: 208
    Labels:
      celltype: 3 strata with value/size (1 (332), 3 (208), 0 (125))
      depth: 3 strata with value/size (0 (125), 1 (332), 2 (208))
      Face Sets: 4 strata with value/size (1 (15), 2 (15), 3 (15), 4 (15))



We can easily verify that the number of ``2-cells`` elements,
i.e. triangles in the mesh has quadrupled. We can also create a new
``MeshMapping`` class to convert the new PETSc ``DMPlex`` into a Netgen
``Mesh`` and visualize it.

.. code:: ipython3

    %%px
    from ngsolve import Mesh
    Map = MeshMapping(newplex)
    from ngsolve.webgui import Draw
    Draw(Mesh(Map.ngMesh))

We can experiment also with other PETSc ``DMPlexTransformation`` for
example the ``REFINETOBOX`` transformation which will split each
triangle in the mesh into three quadrilateral by joining the midpoints
of each edge. This will allow to obtain a purely quadrilateral mesh from
a Netgen mesh.

.. code:: ipython3

    %%px
    tr = PETSc.DMPlexTransform().create(comm=PETSc.COMM_WORLD)
    tr.setType(PETSc.DMPlexTransformType.REFINETOBOX)
    tr.setDM(Map.petscPlex)
    tr.setUp()
    newplex = tr.apply(Map.petscPlex)
    Map = MeshMapping(newplex)
    Draw(Mesh(Map.ngMesh))

Example (Alfeld Splittings and Scott-Vogelious)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example we would like to show that the finite element pair
:math:`P^2-P^1_{disc}` known as the low-order Scott-Vogelious pair,
which is known to verify the Brezzi-Babuska condition for the Stokes
problem on Alfeld split mesh, can be easily implemented in NGSolve using
a ``PETSc DMPlexTransformation``. First we construct a mesh and use the
``MeshMapping`` class to obtain an PETSc ``DMPlex`` that we proceed to
split to obtain an Alfeld refinement.

.. code:: ipython3

    %%px
    from netgen.geom2d import SplineGeometry
    if COMM_WORLD.rank == 0:
        geo = SplineGeometry()
        geo.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
        geo.AddCircle ( (0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
        mesh = Mesh( geo.GenerateMesh(maxh=0.05))
        mesh.Curve(3)
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
    Map = MeshMapping(mesh)
    tr = PETSc.DMPlexTransform().create(comm=PETSc.COMM_WORLD)
    tr.setType(PETSc.DMPlexTransformType.REFINEALFELD)
    tr.setDM(Map.petscPlex)
    tr.setUp()
    newplex = tr.apply(Map.petscPlex)
    mesh = Mesh(MeshMapping(newplex).ngMesh)
    Draw(mesh)

We now proceed constructing the finite element space we are interested
in and distressing the Stokes equation in variational form, i.e. find
:math:`(\vec{u}_h,p_h)\in [P^2(\mathcal{T}_h)]^2\times P^1_{disc}(\mathcal{T}_h)`
such that for any
:math:`(\vec{v}_h,q_h)\in [P^2(\mathcal{T}_h)]^2\times P^1_{disc}(\mathcal{T}_h)`
the follwing equations hold,

.. math::


   (\nabla \vec{u}_h,\nabla \vec{v}_h)-(\nabla \cdot \vec{v}_h,p_h) = (\vec{f},\vec{v}_h)\\
   (\nabla \cdot \vec{u}_h,p_h) = (\vec{f},\vec{v}_h)\\

.. code:: ipython3

    %%px
    from ngsolve import VectorH1, L2, H1, BilinearForm, InnerProduct, GridFunction
    from ngsolve import grad, div, x,y, dx, CoefficientFunction, Norm, SetVisualization, TRIG
    V = VectorH1(mesh, order=2,dirichlet=[4,1,3,5,6,7,8])
    Q = L2(mesh, order=1)
    X = V*Q
    
    u,p = X.TrialFunction()
    v,q = X.TestFunction()
    
    a = BilinearForm(X)
    a += (InnerProduct(grad(u),grad(v))+div(u)*q+div(v)*p)*dx
    a.Assemble()
    
    gfu = GridFunction(X)
    uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
    gfu.components[0].Set(uin, definedon=mesh.Boundaries([3]))
    
    res = gfu.vec.CreateVector()
    res.data = -a.mat * gfu.vec
    inv = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="umfpack")
    gfu.vec.data += inv * res
    Draw(Norm(gfu.components[0]), mesh, "|vel|")
    SetVisualization(max=2)
