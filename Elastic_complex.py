from ufl import *
from dolfin import *
from multiphenics import *
import matplotlib.pyplot as plt
import numpy


a=0.1;
t=a/10;

####PML definition
############################################################
def compProd(a1,b1,a2,b2):
        x1=a1*a2-b1*b2;
        x2=a1*b2+a2*b1;
        return (x1, x2)

class LR(UserExpression):
    def eval(self, value, x):
        a=0.1;t=a/10;

        b1 = numpy.abs(x[0])-a/2; f1=10*(b1+numpy.abs(b1))/(2*t);
        b2 = numpy.abs(x[1])-a/2; f2=10*(b2+numpy.abs(b2))/(2*t);
        l1R=1+f1;l1I=-f1;
        l2R=1+f2;l2I=-f2;

        value[0] = l1R/(pow(l1R,2)+pow(l1I,2));
        value[3] = l2R/(pow(l2R,2)+pow(l2I,2));
        value[1] = 0;
        value[2] = 0;

    def value_shape(self):
        return (2,2)

class LI(UserExpression):
    def eval(self, value, x):
        a=0.1;t=a/10;

        b1 = numpy.abs(x[0])-a/2; f1=10*(b1+numpy.abs(b1))/(2*t);
        b2 = numpy.abs(x[1])-a/2; f2=10*(b2+numpy.abs(b2))/(2*t);
        l1R=1+f1;l1I=-f1;
        l2R=1+f2;l2I=-f2;

        value[0] = -l1I/(pow(l1R,2)+pow(l1I,2));
        value[3] = -l2I/(pow(l2R,2)+pow(l2I,2));
        value[1] = 0;
        value[2] = 0;

    def value_shape(self):
        return (2,2)


class lR(UserExpression):
    def eval(self, value, x):
        a=0.1;t=a/10;

        b1 = numpy.abs(x[0])-a/2; f1=10*(b1+numpy.abs(b1))/(2*t);
        b2 = numpy.abs(x[1])-a/2; f2=10*(b2+numpy.abs(b2))/(2*t);
        l1R=1+f1;l1I=-f1;
        l2R=1+f2;l2I=-f2;
        value[0], temp = compProd(l1R,l1I,l2R,l2I);
#    def value_shape(self):
#        return (1,)

class lI(UserExpression):
    def eval(self, value, x):
        a=0.1;t=a/10;

        b1 = numpy.abs(x[0])-a/2; f1=10*(b1+numpy.abs(b1))/(2*t);
        b2 = numpy.abs(x[1])-a/2; f2=10*(b2+numpy.abs(b2))/(2*t);
        l1R=1+f1;l1I=-f1;
        l2R=1+f2;l2I=-f2;
        temp, value[0] = compProd(l1R,l1I,l2R,l2I);
#    def value_shape(self):
#        return (1,)

class Middle(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0) and between(x[0], (-1/1500, 1/1500)))
    
def OuterBoundaries(x, on_boundary):
    return on_boundary

LambdaR = LR()
LambdaI = LI()
LLR = lR()
LLI = lI()


####Geometry and mesh
############################################################
nx, ny = 200, 200

mesh = RectangleMesh(Point(-a/2-t, -a/2-t), Point(a/2+t, a/2+t), nx, ny)
boundaries = MeshFunction("size_t", mesh, 1)
boundaries.set_all(0)

Middle().mark(boundaries, 1)


####Material properties
############################################################
E, rho, nu = 300E+8, 8000, 0.3
frequency=50000
omega=2*numpy.pi*frequency

lam = E*nu/((1+nu)*(1-2*nu))
mu = E/(2*(1+nu))
i,j,k,l = indices(4)
delta = Identity(2)

C=as_tensor((lam*(delta[i,j]*delta[k,l])+mu*(delta[i,k]*delta[j,l]+delta[i,l]*delta[j,k])),(i,j,k,l))


####Space of functions
############################################################
V = VectorFunctionSpace(mesh, "CG", 1)
Vcomplex = BlockFunctionSpace([V, V], restrict=[None, None])


#Applying the Boundary conditions
###########################################################
BC1 = DirichletBC(Vcomplex.sub(0), Constant((1.0, 0.0)), boundaries, 1)
BC2 = DirichletBC(Vcomplex.sub(1), Constant((0.0, 0.0)), boundaries, 1)
#BC3 = DirichletBC(Vcomplex.sub(0), Constant((0.0, 0.0)), OuterBoundaries)
#BC4 = DirichletBC(Vcomplex.sub(1), Constant((0.0, 0.0)), OuterBoundaries)
bcs = BlockDirichletBC([BC1, BC2])


#Real and Imaginary parts of the trial and test functions
###########################################################
(uR, uI) = BlockTrialFunction(Vcomplex)
(wR, wI) = BlockTestFunction(Vcomplex)


#Weak form and solve
###########################################################
strainR=0.5*(as_tensor((Dx(uR[i],k)*LambdaR[k,j]),(i,j))  +  as_tensor((Dx(uR[k],i)*LambdaR[j,k]),(i,j)))  -  0.5*(as_tensor((Dx(uI[i],k)*LambdaI[k,j]),(i,j))  +  as_tensor((Dx(uI[k],i)*LambdaI[j,k]),(i,j)))
strainI=0.5*(as_tensor((Dx(uR[i],k)*LambdaI[k,j]),(i,j))  +  as_tensor((Dx(uR[k],i)*LambdaI[j,k]),(i,j)))  +  0.5*(as_tensor((Dx(uI[i],k)*LambdaR[k,j]),(i,j))  +  as_tensor((Dx(uI[k],i)*LambdaR[j,k]),(i,j)))

tempR=LLR*LambdaR-LLI*LambdaI
tempI=LLR*LambdaI+LLI*LambdaR
sR=as_tensor((C[i,j,k,l]*strainR[k,l]),[i,j])
sI=as_tensor((C[i,j,k,l]*strainI[k,l]),[i,j])
stressR=as_tensor((sR[i,j]*tempR[j,k]),(i,k))  -  as_tensor((sI[i,j]*tempI[j,k]),(i,k))
stressI=as_tensor((sR[i,j]*tempI[j,k]),(i,k))  +  as_tensor((sI[i,j]*tempR[j,k]),(i,k))


a_11 =  omega**2*rho*dot(LLR*uR,wR)*dx
a_12 = -omega**2*rho*dot(LLI*uI,wR)*dx
a_13 = -inner(stressR, grad(wR))*dx
a_14 =  0
a_21 =  omega**2*rho*dot(LLI*uR,wI)*dx
a_22 = -omega**2*rho*dot(LLR*uI,wI)*dx
a_23 =  0
a_24 = -inner(stressI, grad(wI))*dx

a = [[a_11, a_12, a_13, a_14],
     [a_21, a_22, a_23, a_24]]

L = [0, 0]
L = BlockForm(L, block_function_space=Vcomplex, block_form_rank=1)

A = block_assemble(a)
F = block_assemble(L)
bcs.apply(A)
bcs.apply(F)

sol = BlockFunction(Vcomplex)
block_solve(A, sol.block_vector(), F, "mumps")
uR, uI = block_split(sol)
plt.figure()
plot(uR)
plt.figure()
plot(uI)
plt.show()

file_uR = File("Elastic/uR.pvd")
file_uI = File("Elastic/uI.pvd")
file_uR << uR
file_uI << uI
