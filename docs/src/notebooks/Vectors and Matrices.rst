PETSc Vec and PETSc Mat
-----------------------

This tutorial will be focused on how to use the PETSc ``KSP`` class to
solve the linear systems that are obtained from a finite element
discretization of a partial differential equation (PDE). In particular,
we will show how to use the ``VectorMapping`` class to map PETSc ``Vec``
to NGSolve vectors and vice versa and the ``Matrix`` class to create a
PETSc ``Mat`` from an NGSolve ``BilinearForm``.

We begin initializing the cluster to the test parallel implementation in
a Jupyter notebook, to do this you need also to start the ipycluster
demon, i.e. ``ipcluster start –engines=MPI -n 4``.

Let’s test if the cluster has by initialized correctly by checking the
size of the ``COMM_WORLD``.

.. code:: ipython3

    from ipyparallel import Cluster
    c = await Cluster().start_and_connect(n=1, activate=True)


.. parsed-literal::

    Starting 1 engines with <class 'ipyparallel.cluster.launcher.LocalEngineSetLauncher'>
    100%|██████████| 1/1 [00:05<00:00,  5.70s/engine]


.. code:: ipython3

    %%px
    from mpi4py.MPI import COMM_WORLD
    COMM_WORLD.Get_size()



.. parsed-literal::

    [0;31mOut[0:1]: [0m1


First we need to construct the distributed mesh that will be used to
define the finite element space that will be used to discretize the PDE
here considered.

.. code:: ipython3

    %%px
    from ngsolve import Mesh
    from netgen.geom2d import unit_square
    
    if COMM_WORLD.rank == 0:
        mesh = Mesh(unit_square.GenerateMesh(maxh=0.2).Distribute(COMM_WORLD))
    else:
        mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))

We now proceed constructing a linear polynomial finite element space,
with :math:`H^1` conformity, and discretize the mass matrix that
represent the :math:`L^2` solar product in the discrete context. We
create a mass matrix to initialize a NGSolve vector corresponding a
``GridFunction`` defined on the finite element space here considered.

.. code:: ipython3

    %%px
    from ngsolve import H1, BilinearForm, dx
    fes = H1(mesh, order=1, dirichlet="left|right|top|bottom")
    u,v = fes.TnT()
    m = BilinearForm(u*v*dx).Assemble()
    M = m.mat
    ngsVec = M.CreateColVector()

We are now ready to create a ``VectorMapping`` that we will first use to
construct PETSc ``Vec`` corresponding to the ``ngsVec`` just
initialized. The only information that the ``VectorMapping`` class needs
is the finite element space corresponding to the vector associated to
the ``GridFunction`` we aim to map, this because the NGSolve ``FESpace``
class contains information about the way the degrees of freedom are
distributed and which degrees of freedom are not constrained by the
bodunary conditions

.. code:: ipython3

    %%px
    from ngsPETSc import VectorMapping
    Map = VectorMapping(fes)
    petscVec = Map.petscVec(ngsVec)
    print("Vector type is {} and it has size {}.".format(petscVec.type,petscVec.size))



.. parsed-literal::

    [stdout:0] Vector type is seq and it has size 37.



We now use the ``Matrix`` class to create a PETSc ``Mat`` from a NGSolve
``BilinearForm``. Once the ``Matrix`` class has been set up, it is
possible to access the corresponding PETSc ``Mat`` object as
``Matrix().mat``. By default, if the communicator world is larger than
one ``mat`` is initialized as a PETSc ``mpiaij`` which is the default
sparse parallel matrix in PETSc, while if the communicator world is one
than ``mat`` is initialized as a PETSc ``seqaij`` which is the default
serial matrix in PETSc. We can also spy inside the matrix using the
``Matrix().view()`` method.

.. code:: ipython3

    %%px
    from ngsPETSc import Matrix
    M = Matrix(m.mat, fes.FreeDofs())
    print("Matrix type is {} and it has size {}.".format(M.mat.type,M.mat.size))
    M.view()



.. parsed-literal::

    [stdout:0] Matrix type is seqaij and it has size (17, 17).
    Mat Object: 1 MPI process
      type: seqaij
    row 0: (0, 0.0224274)  (1, 0.00348275)  (10, 0.00389906)  (12, 0.00344571) 
    row 1: (0, 0.00348275)  (1, 0.0198054)  (2, 0.00329403)  (11, 0.00318985)  (12, 0.00335232) 
    row 2: (1, 0.00329403)  (2, 0.0213869)  (3, 0.00364805)  (11, 0.00320442) 
    row 3: (2, 0.00364805)  (3, 0.0187914)  (4, 0.00276012)  (11, 0.0029006)  (15, 0.00252874) 
    row 4: (3, 0.00276012)  (4, 0.0161198)  (5, 0.00273449)  (13, 0.0025502)  (15, 0.00239586) 
    row 5: (4, 0.00273449)  (5, 0.0170459)  (6, 0.00262495)  (13, 0.00256258) 
    row 6: (5, 0.00262495)  (6, 0.0125646)  (7, 0.00238756)  (13, 0.00238142) 
    row 7: (6, 0.00238756)  (7, 0.0176799)  (8, 0.00325374)  (13, 0.00285307)  (16, 0.00338652) 
    row 8: (7, 0.00325374)  (8, 0.0200592)  (9, 0.00312461)  (14, 0.00350041)  (16, 0.00368234) 
    row 9: (8, 0.00312461)  (9, 0.0189321)  (14, 0.00362765) 
    row 10: (0, 0.00389906)  (10, 0.0211063)  (12, 0.00340267)  (14, 0.00383436) 
    row 11: (1, 0.00318985)  (2, 0.00320442)  (3, 0.0029006)  (11, 0.0187766)  (12, 0.0034216)  (15, 0.0027623)  (16, 0.00329787) 
    row 12: (0, 0.00344571)  (1, 0.00335232)  (10, 0.00340267)  (11, 0.0034216)  (12, 0.0210307)  (14, 0.00364803)  (16, 0.00376036) 
    row 13: (4, 0.0025502)  (5, 0.00256258)  (6, 0.00238142)  (7, 0.00285307)  (13, 0.0159034)  (15, 0.00253607)  (16, 0.0030201) 
    row 14: (8, 0.00350041)  (9, 0.00362765)  (10, 0.00383436)  (12, 0.00364803)  (14, 0.0226675)  (16, 0.00387176) 
    row 15: (3, 0.00252874)  (4, 0.00239586)  (11, 0.0027623)  (13, 0.00253607)  (15, 0.0130458)  (16, 0.00282287) 
    row 16: (7, 0.00338652)  (8, 0.00368234)  (11, 0.00329787)  (12, 0.00376036)  (13, 0.0030201)  (14, 0.00387176)  (15, 0.00282287)  (16, 0.0238418) 



There are other matrices format that are wrapped some of which are
device dependent, to mention a few: - ``dense``, store and operate on
the matrix in dense format, - ``cusparse``, store and operate on the
matrix on NVIDIA GPU device in CUDA sparse format, - ``aijmkl``, store
and operate on the matrix in Intel MKL format.

.. code:: ipython3

    %%px
    M = Matrix(m.mat, fes.FreeDofs(), matType="dense")

Example (Precondition Inverse Iteration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We here implement the Precondition INVerse ITeration (PINVIT) developed
by Knyazef and Neymeyr, more detail
`here <https://doi.org/10.1016/S0024-3795(00)00239-1>`__, using PETSc.
In particular, we will use the PINVIT scheme to compute the eigenvalue
of the Laplacian, i.e. we are looking for :math:`\lambda\in \mathbb{R}`
such that it exits :math:`u\in H^1_0(\Omega)` that verifies following
equation for any :math:`v\in H^1_0(\Omega)`

.. math:: \int_\Omega \nabla u \cdot \nabla v \; d\vec{x} = \lambda \int_\Omega uv\;d\vec{x}

We solve this specific problem by looking for the eigenvalue of the
generalised eigenproblem :math:`A\vec{u}_h = \lambda M\vec{u}_h` where
:math:`A` and :math:`M` are the finite element discretisation
respectively of the stifness matrix corresponding to the Laplacian and
the mass matrix corresponding to the :math:`L^2` inner prodcut. We begin
constructin the finite element discretisation for :math:`A` and
:math:`M`.

.. code:: ipython3

    %%px
    from ngsolve import grad, Preconditioner, GridFunction
    a = BilinearForm(fes)
    a += grad(u)*grad(v)*dx
    pre = Preconditioner(a, "multigrid")
    a.Assemble()
    u = GridFunction(fes)

The heart of the PINVIT scheme there is an iteration similar idea to the
Rayleigh quotient iteration for a generalised eigenvalue problem, more
detail can be found in Nick Trefethen’s `Numerical Linear
Algebra <https://doi.org/10.1137/1.9780898719574>`__, Lecture 27:

.. math:: \vec{u}_h^{(n+1)} = \omega_1^{(n)}\vec{u}_{h}^{(n)}+\omega_2^{(n)} \vec{\omega}_h^{(n)}, \qquad \vec{\omega}_h^{(n)}= P^{-1}(A\vec{u}_h^{(n)}-\rho_n M\vec{u}_h^{(n)}),

where :math:`P^{-1}` is an approximate inverse of the stifness matrix
:math:`A` and :math:`\rho_n` is the Rayleigh quotient corresponding to
:math:`\vec{u}_h^{(n)}`, i.e.

.. math:: \rho_{n} = \frac{(\vec{u}_h^{(n)}, A \vec{u}_h^{(n)})}{(\vec{u}_h^{(n)}, M\vec{u}_h^{(n)})}.

Instrumental in order to obtain a converged PINVIT scheme is our choice
of :math:`\alpha_n`, but we will postpone this discuss and first
implement the previous itration for a fixed choice of
:math:`\omega_i^{(n)}`.

.. code:: ipython3

    %%px
    def stepChoice(Asc,Msc,w,u0):
        return (0.5,0.5)

We begin constructing a PETSc ``Mat`` object corresponding to :math:`A`
and :math:`M` using the ngsPETSc ``Metrix`` class. We then construct a
``VectorMapping`` to object to convert NGSolve ``GridFunction`` to PETSc
``Vec``.

.. code:: ipython3

    %%px
    A = Matrix(a.mat, fes.FreeDofs())
    M = Matrix(m.mat, fes.FreeDofs())
    Map = VectorMapping(fes)

We then construct a PETSc ``PC`` object used to create an approximate
inverse of :math:`A`, in particular we will be interested in using a
preconditioner build using HYPRE.

.. code:: ipython3

    %%px
    from petsc4py import PETSc
    pc = PETSc.PC()
    pc.create(PETSc.COMM_WORLD)
    pc.setOperators(A.mat)
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setUp()

We now implement the iteration itself, starting from a PETSc ``Vec``
that we create from a PETSc ``Mat`` to be sure it has the correct size,
and that we then set to have random entries.

.. code:: ipython3

    %%px
    from math import pi
    itMax = 10
    u0 = A.mat.createVecLeft()
    w = A.mat.createVecLeft()
    u0.setRandom()
    for it in range(itMax):
            Au0 = u0.duplicate(); A.mat.mult(u0,Au0)
            Mu0 = u0.duplicate(); M.mat.mult(u0,Mu0)
            rho = Au0.dot(u0)/Mu0.dot(u0)
            print("[{}] Eigenvalue estimate: {}".format(it,rho/(pi**2)))
            u = Au0+rho*Mu0
            pc.apply(u,w)
            alpha = stepChoice(A.mat,M.mat,w,u0)
            u0 = alpha[0]*u0+alpha[1]*w



.. parsed-literal::

    [stdout:0] [0] Eigenvalue estimate: 6.438964160408917
    [1] Eigenvalue estimate: 3.343928625687638
    [2] Eigenvalue estimate: 2.641648954164419
    [3] Eigenvalue estimate: 2.3821542621098084
    [4] Eigenvalue estimate: 2.2665070692423788
    [5] Eigenvalue estimate: 2.210409247274843
    [6] Eigenvalue estimate: 2.1819357667646018
    [7] Eigenvalue estimate: 2.167055475874936
    [8] Eigenvalue estimate: 2.15909469308555
    [9] Eigenvalue estimate: 2.1547369780738275



We now need to discuss how to choose the step size :math:`\omega_i` and
we do this by solving the optimization problem,

.. math:: \vec{u}_h^{(n+1)} = \underset{\vec{v}\in <\vec{u}_h^{n},\, \vec{\omega}_h^{(n)}>}{arg\;min} \frac{(\vec{u}_h^{(n+1)}, A \vec{u}_h^{(n+1)})}{(\vec{u}_h^{(n+1)}, M\vec{u}_h^{(n+1)})}

and we do solving a small generalised eigenvalue problem, i.e.

.. math::

   \begin{bmatrix}
   \vec{u}_h^{(n)}\cdot A \vec{u}_h^{(n)} & \vec{u_h}^{(n)}\cdot A \vec{\omega}_h^{(n)}\\
   \vec{\omega}_h^{(n)}\cdot A \vec{u}_h^{(n)} & \vec{\omega}_h^{(n)}\cdot A \vec{\omega}_h^{(n)}
   \end{bmatrix} = \omega \begin{bmatrix}
   \vec{u}_h^{(n)}\cdot M \vec{u}_h^{(n)} & \vec{u_h}^{(n)}\cdot M \vec{\omega}_h^{(n)}\\
   \vec{\omega}_h^{(n)}\cdot M \vec{u}_h^{(n)} & \vec{\omega}_h^{(n)}\cdot M \vec{\omega}_h^{(n)}
   \end{bmatrix}.

.. code:: ipython3

    %%px
    import numpy as np
    from scipy.linalg import eigh
    def stepChoice(Asc,Msc,w,u0):
        Au0 = u0.duplicate(); Asc.mult(u0,Au0)
        Mu0 = u0.duplicate(); Msc.mult(u0,Mu0)
        Aw = w.duplicate(); Asc.mult(w,Aw)
        Mw = w.duplicate(); Msc.mult(w,Mw)
        smallA = np.array([[u0.dot(Au0),u0.dot(Aw)],[w.dot(Au0),w.dot(Aw)]])
        smallM = np.array([[u0.dot(Mu0),u0.dot(Mw)],[w.dot(Mu0),w.dot(Mw)]])
        _, evec = eigh(a=smallA, b=smallM)
        return (float(evec[0,0]),float(evec[1,0]))
    
    itMax = 10
    u0 = A.mat.createVecLeft()
    w = A.mat.createVecLeft()
    u0.setRandom()
    for it in range(itMax):
            Au0 = u0.duplicate(); A.mat.mult(u0,Au0)
            Mu0 = u0.duplicate(); M.mat.mult(u0,Mu0)
            rho = Au0.dot(u0)/Mu0.dot(u0)
            print("[{}] Eigenvalue estimate: {}".format(it,rho/(pi**2)))
            u = Au0+rho*Mu0
            pc.apply(u,w)
            alpha = stepChoice(A.mat,M.mat,w,u0)
            u0 = alpha[0]*u0+alpha[1]*w



.. parsed-literal::

    [stdout:0] [0] Eigenvalue estimate: 6.438964160408917
    [1] Eigenvalue estimate: 2.182148561544114
    [2] Eigenvalue estimate: 2.1494909780380205
    [3] Eigenvalue estimate: 2.148207487071055
    [4] Eigenvalue estimate: 2.1481654601579416
    [5] Eigenvalue estimate: 2.1481654570280586
    [6] Eigenvalue estimate: 2.148165457028058
    [7] Eigenvalue estimate: 2.1481654570280573
    [8] Eigenvalue estimate: 2.1481654570280577
    [9] Eigenvalue estimate: 2.1481654570280577


