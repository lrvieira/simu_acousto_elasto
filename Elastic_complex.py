from ufl import *
from dolfin import *
from multiphenics import *
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


class lR(Expression):
    def eval(self, value, x):
        a=0.1;t=a/10;

        b1 = numpy.abs(x[0])-a/2; f1=10*(b1+numpy.abs(b1))/(2*t);
        b2 = numpy.abs(x[1])-a/2; f2=10*(b2+numpy.abs(b2))/(2*t);
        l1R=1+f1;l1I=-f1;
        l2R=1+f2;l2I=-f2;
        value[0], temp = compProd(l1R,l1I,l2R,l2I);
#    def value_shape(self):
#        return (1,)

class lI(Expression):
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
        return (near(x[1], 0) and between(x[0], (-1,1)))

LambdaR = LR()
LambdaI = LI()
LLR = lR
LLI = lI


####Geometrical properties
############################################################
x_1, y_1 = -6.0, -4.0
x_2, y_2 = 6.0, 4.0
nx, ny = 300, 200

mesh = RectangleMesh(Point(x_1, y_1), Point(x_2, y_2), nx, ny)
boundaries = MeshFunction("size_t", mesh, 1)
boundaries.set_all(0)

Middle().mark(boundaries, 1)


####Material propertie
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
source = [DirichletBC(Vcomplex.sub(0), Constant((1.0, 0.0)), boundaries, 1), DirichletBC(Vcomplex.sub(1), Constant((1.0, 0.0)), boundaries, 1)]
bc=source


#Real and Imaginary parts of the trial and test functions
###########################################################
(uR, uI) = BlockTrialFunction(Vcomplex)
(wR, wI) = BlockTestFunction(Vcomplex)


strainR=0.5*(as_tensor((Dx(uR[i],k)*LambdaR[k,j]),(i,j,k))  +  as_tensor((Dx(uR[k],i)*LambdaR[j,k]),(i,j)))  -  0.5*(as_tensor((Dx(uI[i],k)*LambdaI[k,j]),(i,j))  +  as_tensor((Dx(uI[k],i)*LambdaI[j,k]),(i,j,k)))
strainI=0.5*(as_tensor((Dx(uR[i],k)*LambdaI[k,j]),(i,j))  +  as_tensor((Dx(uR[k],i)*LambdaI[j,k]),(i,j)))  +  0.5*(as_tensor((Dx(uI[i],k)*LambdaR[k,j]),(i,j))  +  as_tensor((Dx(uI[k],i)*LambdaR[j,k]),(i,j)))


tempR=LLR*LambdaR-LLI*LambdaI
tempI=LLR*LambdaI+LLI*LambdaR
sR=as_tensor((C[i,j,k,l]*strainR[k,l]),[i,j])
sI=as_tensor((C[i,j,k,l]*strainI[k,l]),[i,j])
stressR=as_tensor((sR[i,j]*tempR[j,k]),(i,k))  -  as_tensor((sI[i,j]*tempI[j,k]),(i,k))
stressI=as_tensor((sR[i,j]*tempI[j,k]),(i,k))  +  as_tensor((sI[i,j]*tempR[j,k]),(i,k))


F=omega**2*rho*(dot((LLR*uR-LLI*uI),wR)+dot((LLR*uI+LLI*uR),wI))*dx-(inner(stressR, grad(wR))+inner(stressI, grad(wI)))*dx
a, L = lhs(F), rhs(F)
