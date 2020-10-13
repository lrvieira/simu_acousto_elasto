from ufl import *
from dolfin import *
from multiphenics import *
import matplotlib.pyplot as plt
import numpy

a=0.1
t=a/10
c = 340
kappa = 250

sigma = 1000
omega = c * kappa
tol = 1E-36

class Source(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0) and between(x[0], (-1/1500, 1/1500)))

nx, ny = 200, 200
mesh = RectangleMesh(Point(-a/2-t, -a/2-t), Point(a/2+t, a/2+t), nx, ny)
boundaries = MeshFunction("size_t", mesh, 1)
boundaries.set_all(0)

Source().mark(boundaries, 1)

VR = FunctionSpace(mesh, "CG", 1)
VI = FunctionSpace(mesh, "CG", 1)
W = BlockFunctionSpace([VR, VI], restrict=[None, None])

one = Constant(1.0)
BC1 = DirichletBC(W.sub(0), one, boundaries, 1)
BC2 = DirichletBC(W.sub(1), one, boundaries, 1)
#BC3 = DirichletBC(W.sub(0), Constant(0.0), boundaries, 2)
#BC4 = DirichletBC(W.sub(1), Constant(0.0), boundaries, 2)
bcs = BlockDirichletBC([BC1, BC2])

(pR, pI) = BlockTrialFunction(W)
(qR, qI) = BlockTestFunction(W)


#yR = [gR_y_x(), gR_x_y(), gR_xy()]
#yI = [gI_y_x(), gI_x_y(), gI_xy()]
yR = [1, 1, 1]
yI = [1, 1, 1]

aF_11 = (yR[0]*pR.dx(0)*qR.dx(0) + yR[1]*pR.dx(1)*qR.dx(1) - kappa**2 * yR[2]*inner(pR, qR))*dx
aF_12 = (yR[0]*pI.dx(0)*qR.dx(0) + yR[1]*pI.dx(1)*qR.dx(1) - kappa**2 * yR[2]*inner(pI, qR))*dx
aF_13 = (yI[0]*pR.dx(0)*qR.dx(0) + yI[1]*pR.dx(1)*qR.dx(1) - kappa**2 * yI[2]*inner(pR, qR))*dx
aF_14 = (yI[0]*pI.dx(0)*qR.dx(0) + yI[1]*pI.dx(1)*qR.dx(1) - kappa**2 * yI[2]*inner(pI, qR))*dx
aF_21 =-(yR[0]*pR.dx(0)*qI.dx(0) + yR[1]*pR.dx(1)*qI.dx(1) - kappa**2 * yR[2]*inner(pR, qI))*dx
aF_23 =-(yI[0]*pR.dx(0)*qI.dx(0) + yI[1]*pR.dx(1)*qI.dx(1) - kappa**2 * yI[2]*inner(pR, qI))*dx
aF_22 = (yR[0]*pI.dx(0)*qI.dx(0) + yR[1]*pI.dx(1)*qI.dx(1) - kappa**2 * yR[2]*inner(pI, qI))*dx
aF_24 = (yI[0]*pI.dx(0)*qI.dx(0) + yI[1]*pI.dx(1)*qI.dx(1) - kappa**2 * yI[2]*inner(pI, qI))*dx

aF = [[aF_11, aF_12, aF_13, aF_14],
      [aF_21, aF_22, aF_23, aF_24]]

LF = [0, 0]
LF = BlockForm(LF, block_function_space=W, block_form_rank=1)

A = block_assemble(aF)
F = block_assemble(LF)
bcs.apply(A)
bcs.apply(F)

sol = BlockFunction(W)
block_solve(A, sol.block_vector(), F, "mumps")
pR, pI = block_split(sol)
plt.figure()
plot(pR)
plt.figure()
plot(pI)
plt.show()
