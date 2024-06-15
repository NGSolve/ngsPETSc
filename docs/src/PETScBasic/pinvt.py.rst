Preconditioned Inverse Iteration for Laplace Eigenvalue Problems
=================================================================
We will here implement a simple preconditioned inverse iteration
using basic PETSc functionalities.
In pparticular, we will show how to use the :code:`VectorMapping` class to
map PETSc vectors to NGSolve vectors and vice versa and the
:code:`Matrix` class to create a PETSc matrix from an NGSolve
:code:`BilinearForm`.
First we need to construct the distributed mesh that will be used to
define the finite element space that will be used to discretize the
PDE here considered. ::

   from ngsolve import Mesh
   from netgen.geom2d import unit_square
   import netgen.meshing as ngm

   if COMM_WORLD.rank == 0:
         mesh = Mesh(unit_square.GenerateMesh(maxh=0.2).Distribute(COMM_WORLD))
   else:
         mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))

We now proceed constructing a linear polynomial finite element space,
with :math:`H^1` conformity, and discretize the mass matrix that
represent the :math:`L^2` scalar product in the discrete context. We
create a mass matrix to initialize an NGSolve vector corresponding to a
:code:`GridFunction` defined on the finite element space here considered. ::

      from ngsolve import H1, BilinearForm, dx
      fes = H1(mesh, order=1, dirichlet="left|right|top|bottom")
      u,v = fes.TnT()
      m = BilinearForm(u*v*dx).Assemble()
      M = m.mat
      ngsVec = M.CreateColVector()

We are now ready to create a :code:`VectorMapping` that we will first use
to construct a PETSc vecotr corresponding to the :code:`ngsVec` just
initialized. The only information that the :code:`VectorMapping` class
needs is the finite element space corresponding to the vector
associated with the :code:`GridFunction` we aim to map, because the
NGSolve :cdoe:`FESpace` class contains information about the way the
degrees of freedom are distributed and which degrees of freedom are
not constrained by the boundary conditions. ::

   from ngsPETSc import VectorMapping
   Map = VectorMapping(fes)
   petscVec = Map.petscVec(ngsVec)
   print("Vector type is {} and it has size {}.".format(petscVec.type,petscVec.size))

We now use the :code:`Matrix` class to create a PETSc matrix from a
NGSolve :code:`BilinearForm`. Once the :code:`Matrix` class has been set up,
it is possible to access the corresponding PETSc matrix object as
:code:`Matrix().mat`. By default, if the communicator world is larger
than one :code:`mat` is initialized as a PETSc ``mpiaij`` which is the
default sparse parallel matrix in PETSc, while if the communicator
world is one then :code:`mat` is initialized as a PETSc ``seqaij`` which
is the default serial matrix in PETSc. We can also spy inside the
matrix using the ``Matrix().view()`` method. ::

   from ngsPETSc import Matrix
   M = Matrix(m.mat, fes)
   print("Matrix type is {} and it has size {}.".format(M.mat.type,M.mat.size))
   M.view()

There are other matrices format that are wrapped some of which are
device dependent, to mention a few:

-  ``dense``, store and operate on the matrix in dense format,
-  ``cusparse``, store and operate on the matrix on NVIDIA GPU device
   in CUDA sparse format,
-  ``aijmkl``, store and operate on the matrix in Intel MKL format.


We will now focus on implementing the Precondition INVerse ITeration (PINVIT)
developed by Knyazef and Neymeyr, more detail
`here <https://doi.org/10.1016/S0024-3795(00)00239-1>`__, using
PETSc. In particular, we will use the PINVIT scheme to compute the
eigenvalue of the Laplacian, i.e. we are looking for
:math:`\lambda\in \mathbb{R}` such that it exits
:math:`u\in H^1_0(\Omega)` that verifies following equation for any
:math:`v\in H^1_0(\Omega)`

   .. math:: \int_\Omega \nabla u \cdot \nabla v \; d\vec{x} = \lambda \int_\Omega uv\;d\vec{x}.

We solve this specific problem by looking for the eigenvalue of the
generalised eigenproblem :math:`A\vec{u}_h = \lambda M\vec{u}_h`
where :math:`A` and :math:`M` are the finite element discretisation
respectively of the stiffness matrix corresponding to the Laplacian
and the mass matrix corresponding to the :math:`L^2` inner product.
We begin constructing the finite element discretisation for :math:`A`
and :math:`M`. ::

   from ngsolve import grad, Preconditioner, GridFunction
   a = BilinearForm(fes)
   a += grad(u)*grad(v)*dx
   a.Assemble()
   u = GridFunction(fes)

The heart of the PINVIT scheme there is an iteration similar idea to
the Rayleigh quotient iteration for a generalised eigenvalue problem,
more detail can be found in Nick Trefethen's `Numerical Linear
Algebra <https://doi.org/10.1137/1.9780898719574>`__, Lecture 27:

   .. math:: \vec{u}_h^{(n+1)} = \omega_1^{(n)}\vec{u}_{h}^{(n)}+\omega_2^{(n)} \vec{\omega}_h^{(n)}, \qquad \vec{\omega}_h^{(n)}= P^{-1}(A\vec{u}_h^{(n)}-\rho_n M\vec{u}_h^{(n)}),

where :math:`P^{-1}` is an approximate inverse of the stifness matrix
:math:`A` and :math:`\rho_n` is the Rayleigh quotient corresponding
to :math:`\vec{u}_h^{(n)}`, i.e.

   .. math:: \rho_{n} = \frac{(\vec{u}_h^{(n)}, A \vec{u}_h^{(n)})}{(\vec{u}_h^{(n)}, M\vec{u}_h^{(n)})}.

Instrumental in order to obtain a converged PINVIT scheme is our
choice of :math:`\alpha_n`, but we will postpone this discussion and
first, implement the previous iteration for a fixed choice of
:math:`\omega_i^{(n)}`. ::

   def stepChoice(Asc,Msc,w,u0):
         return (0.5,0.5)

We begin constructing a PETSc matrix object corresponding to
:math:`A` and :math:`M` using the ngsPETSc ``Matrix`` class. We then
construct a :code:`VectorMapping` to convert NGSolve :code:`GridFunction` to
PETSc vectors. ::

   A = Matrix(a.mat, fes)
   M = Matrix(m.mat, fes)
   Map = VectorMapping(fes)

We then construct a PETSc preconditioner object used to create an approximate
inverse of :math:`A`, in particular we will be interested in using a
preconditioner build using HYPRE. ::

      from petsc4py import PETSc
      pc = PETSc.PC()
      pc.create(PETSc.COMM_WORLD)
      pc.setOperators(A.mat)
      pc.setType(PETSc.PC.Type.HYPRE)
      pc.setUp()

We now implement the iteration itself, starting from a PETSc vector
that we create from a PETSc matrix to be sure it has the correct
size, and that we then set to have random entries. ::

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

We now need to discuss how to choose the step size :math:`\omega_i`
and we do this by solving the optimization problem,

   .. math:: \vec{u}_h^{(n+1)} = \underset{\vec{v}\in <\vec{u}_h^{n},\, \vec{\omega}_h^{(n)}>}{arg\;min} \frac{(\vec{u}_h^{(n+1)}, A \vec{u}_h^{(n+1)})}{(\vec{u}_h^{(n+1)}, M\vec{u}_h^{(n+1)})}

and we do solve a small generalized eigenvalue problem, i.e.

   .. math::

      \begin{bmatrix}
      \vec{u}_h^{(n)}\cdot A \vec{u}_h^{(n)} & \vec{u_h}^{(n)}\cdot A \vec{\omega}_h^{(n)}\\
      \vec{\omega}_h^{(n)}\cdot A \vec{u}_h^{(n)} & \vec{\omega}_h^{(n)}\cdot A \vec{\omega}_h^{(n)}
      \end{bmatrix} = \omega \begin{bmatrix}
      \vec{u}_h^{(n)}\cdot M \vec{u}_h^{(n)} & \vec{u_h}^{(n)}\cdot M \vec{\omega}_h^{(n)}\\
      \vec{\omega}_h^{(n)}\cdot M \vec{u}_h^{(n)} & \vec{\omega}_h^{(n)}\cdot M \vec{\omega}_h^{(n)}
      \end{bmatrix}.

::

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
