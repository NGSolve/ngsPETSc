Preconditioned Inverse Iteration for Laplace Eigenvalue Problems
=================================================================
We will use the Preconditioned INVerse ITeration (PINVIT) scheme, 
developed by `Knyazef and Neymeyr <https://doi.org/10.1016/S0024-3795(00)00239-1>`__,
to compute the lowest eigenvalues of the Dirichlet Laplacian. We seek
:math:`(u,\lambda) \in H^1_0(\Omega)\times \mathbb{R}` such that for any
:math:`v\in H^1_0(\Omega)`

   .. math:: \int_\Omega \nabla u \cdot \nabla v \; d\vec{x} = \lambda \int_\Omega uv\;d\vec{x}.

In the process of coding the PINVIT scheme, we will show how to use the :code:`VectorMapping` class to
map PETSc vectors to NGSolve vectors and vice versa. We will also show how to use the :code:`Matrix` class 
to create a PETSc matrix from an NGSolve :code:`BilinearForm`.
First, we need to construct the distributed mesh for the finite element space discretising the PDE. ::

   from ngsolve import Mesh, unit_square
   import netgen.meshing as ngm
   import numpy as np
   from scipy.linalg import eigh
   from mpi4py.MPI import COMM_WORLD

   mesh = Mesh(unit_square.GenerateMesh(maxh=0.2, comm=COMM_WORLD))

We now proceed to construct a linear polynomial finite element space, with :math:`H^1` conformity, 
and discretise the mass matrix that represents the :math:`L^2` inner product. We also fetch a 
compatible vector from the mass matrix. ::

   from ngsolve import H1, BilinearForm, dx
   fes = H1(mesh, order=1, dirichlet="left|right|top|bottom")
   u,v = fes.TnT()
   m = BilinearForm(u*v*dx).Assemble()
   M = m.mat
   ngsVec = M.CreateColVector()

We are now ready to create a :code:`VectorMapping`, which will first be used to construct 
a PETSc vector from the :code:`ngsVec` just initialized.
The only information that the :code:`VectorMapping` class needs is the finite element space
associated to the vector :code:`GridFunction`, this because the NGSolve :code:`FESpace` class 
contains information about the way the degrees of freedom are distributed and which degrees of
freedom are not constrained by the boundary conditions. ::

   from ngsPETSc import VectorMapping
   Map = VectorMapping(fes)
   petscVec = Map.petscVec(ngsVec)
   print("Vector type is {} and it has size {}.".format(petscVec.type,petscVec.size))

We now use the :code:`Matrix` class to create a PETSc matrix from an NGSolve :code:`BilinearForm`. 
Once the :code:`Matrix` class has been set up, it is possible to access the corresponding PETSc matrix
object as :code:`Matrix().mat`. By default, if :code:`COMM_WORLD.GetSize()` is larger than one, 
:code:`mat` is initialized as a PETSc ``mpiaij`` which is the default sparse parallel matrix in PETSc, 
otherwise :code:`mat` is initialized as a PETSc ``seqaij`` which is the default serial matrix in PETSc.
We can also spy inside the matrix using the :code:`Matrix().view()` method. ::

   from ngsPETSc import Matrix
   M = Matrix(m.mat, fes)
   print("Matrix type is {} and it has size {}.".format(M.mat.type,M.mat.size))
   M.view()

There are other matrix formats aveilable. To mention a few:

-  ``dense``, a dense format,
-  ``cusparse``, CUDA sparse format, for NVIDIA GPU.
-  ``aijmkl``, Intel MKL format.


We solve the discretised problem by looking for the eigenvalues of the generalised eigenproblem
:math:`A\vec{u}_h = \lambda M\vec{u}_h` where :math:`A` and :math:`M` are the finite element discretisations
respectively of the stiffness matrix for the Laplacian and the mass matrix.
Since we have already constructed the mass matrix, it remains to construct the stiffness matrix: ::

   from ngsolve import grad, Preconditioner, GridFunction
   a = BilinearForm(fes)
   a += grad(u)*grad(v)*dx
   a.Assemble()
   A = Matrix(a.mat, fes)

At the heart of the PINVIT scheme there is an iteration similar to the Rayleigh quotient iteration
for a generalised eigenvalue problem. More details on the latter can be found in Nick Trefethen's
`Numerical Linear Algebra <https://doi.org/10.1137/1.9780898719574>`__, Lecture 27:

   .. math:: \vec{u}_h^{(n+1)} = \omega_1^{(n)}\vec{u}_{h}^{(n)}+\omega_2^{(n)} \vec{\omega}_h^{(n)}, \qquad \vec{\omega}_h^{(n)}= P^{-1}(A\vec{u}_h^{(n)}-\rho_n M\vec{u}_h^{(n)}),

where :math:`P^{-1}` is an approximate inverse of the stiffness matrix :math:`A`, :math:`\omega_i` are
step sizes, and :math:`\rho_n` is the Rayleigh quotient corresponding to :math:`\vec{u}_h^{(n)}`, i.e.

   .. math:: \rho_{n} = \frac{(\vec{u}_h^{(n)}, A \vec{u}_h^{(n)})}{(\vec{u}_h^{(n)}, M\vec{u}_h^{(n)})}.

The choice of :math:`\omega_{1,2}^{(n)}` is important to obtain a convergent PINVIT scheme.
We do this by solving the optimization problem,

   .. math:: \vec{u}_h^{(n+1)} = \underset{\vec{v}\in <\vec{u}_h^{(n)},\, \vec{\omega}_h^{(n)}>}{arg\;min} \frac{(\vec{u}_h^{(n+1)}, A \vec{u}_h^{(n+1)})}{(\vec{u}_h^{(n+1)}, M\vec{u}_h^{(n+1)})}

which is equivalent to a small generalised eigenvalue problem, i.e.

   .. math::
      \begin{bmatrix}
      \vec{u}_h^{(n)}\cdot A \vec{u}_h^{(n)} & \vec{u_h}^{(n)}\cdot A \vec{\omega}_h^{(n)}\\
      \vec{\omega}_h^{(n)}\cdot A \vec{u}_h^{(n)} & \vec{\omega}_h^{(n)}\cdot A \vec{\omega}_h^{(n)}
      \end{bmatrix} = \omega \begin{bmatrix}
      \vec{u}_h^{(n)}\cdot M \vec{u}_h^{(n)} & \vec{u_h}^{(n)}\cdot M \vec{\omega}_h^{(n)}\\
      \vec{\omega}_h^{(n)}\cdot M \vec{u}_h^{(n)} & \vec{\omega}_h^{(n)}\cdot M \vec{\omega}_h^{(n)}
      \end{bmatrix}.

::

   def stepChoice(Asc,Msc,w,u0):
         Au0 = u0.duplicate(); Asc.mult(u0,Au0)
         Mu0 = u0.duplicate(); Msc.mult(u0,Mu0)
         Aw = w.duplicate(); Asc.mult(w,Aw)
         Mw = w.duplicate(); Msc.mult(w,Mw)
         smallA = np.array([[u0.dot(Au0),u0.dot(Aw)],[w.dot(Au0),w.dot(Aw)]])
         smallM = np.array([[u0.dot(Mu0),u0.dot(Mw)],[w.dot(Mu0),w.dot(Mw)]])
         _, evec = eigh(a=smallA, b=smallM)
         return (float(evec[0,0]),float(evec[1,0]))



We then construct a PETSc preconditioner acting as an approximate inverse of :math:`A`.
For this example, we use an algebraic multigrid preconditioner built with HYPRE. ::

   from petsc4py import PETSc
   pc = PETSc.PC()
   pc.create(PETSc.COMM_WORLD)
   pc.setOperators(A.mat)
   pc.setType(PETSc.PC.Type.HYPRE)
   pc.setUp()

We now implement the iteration itself: :: 

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