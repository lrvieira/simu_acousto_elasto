from ufl import *
from dolfin import *
from multiphenics import *
import matplotlib.pyplot as plt
import numpy


#========== PML, PHYSICAL CONSTANTS AND GEOMETRY DEFINITIONS
#===================================================================================

a=0.1
t=a/10
c = 340
kappa = 800

sigma = 250
omega = c*kappa
tol = 1E-14


#========== GAMMA FUNCTIONS (gamma_y/gamma_x etc.)
#===================================================================================

class gR_y_x(UserExpression):
    def eval(self, value, x):
        sigma_x = sigma*(numpy.abs(x[0])-a/2)**2
        sigma_y = sigma*(numpy.abs(x[1])-a/2)**2
        
        if numpy.abs(x[0]) > a/2 + tol and numpy.abs(x[1]) > a/2 + tol:
            value[0] = (omega**2 + sigma_y*sigma_x)/(omega**2 + sigma_x**2)
        
        elif numpy.abs(x[0]) > a/2 + tol and numpy.abs(x[1]) < a/2 + tol:
            value[0] = (omega**2)/(omega**2 + sigma_x**2)
            
        elif numpy.abs(x[0]) < a/2 + tol and numpy.abs(x[1]) > a/2 + tol:
            value[0] = 1
            
        else:
            value[0] = 1
            
    def value_shape(self):
        return ()
            
            
class gI_y_x(UserExpression):
    def eval(self, value, x):
        sigma_x = sigma*(numpy.abs(x[0])-a/2)**2
        sigma_y = sigma*(numpy.abs(x[1])-a/2)**2
        
        if numpy.abs(x[0]) > a/2 + tol and numpy.abs(x[1]) > a/2 + tol:
            value[0] = omega*(sigma_y - sigma_x)/(omega**2 + sigma_x**2)
        
        elif numpy.abs(x[0]) > a/2 + tol and numpy.abs(x[1]) < a/2 + tol:
            value[0] = -(omega*sigma_x)/(omega**2 + sigma_x**2)
            
        elif numpy.abs(x[0]) < a/2 + tol and numpy.abs(x[1]) > a/2 + tol:
            value[0] = sigma_y/omega
            
        else:
            value[0] = 0
            
    def value_shape(self):
        return ()
            

class gR_x_y(UserExpression):
    def eval(self, value, x):
        sigma_x = sigma*(numpy.abs(x[0])-a/2)**2
        sigma_y = sigma*(numpy.abs(x[1])-a/2)**2
        
        if numpy.abs(x[0]) > a/2 + tol and numpy.abs(x[1]) > a/2 + tol:
            value[0] = (omega**2 + sigma_x*sigma_y)/(omega**2 + sigma_y**2)
        
        elif numpy.abs(x[0]) > a/2 + tol and numpy.abs(x[1]) < a/2 + tol:
            value[0] = 1
            
        elif numpy.abs(x[0]) < a/2 + tol and numpy.abs(x[1]) > a/2 + tol:
            value[0] = (omega**2)/(omega**2 + sigma_y**2)
            
        else:
            value[0] = 1
            
    def value_shape(self):
        return ()


class gI_x_y(UserExpression):
    def eval(self, value, x):
        sigma_x = sigma*(numpy.abs(x[0])-a/2)**2
        sigma_y = sigma*(numpy.abs(x[1])-a/2)**2
        
        if numpy.abs(x[0]) > a/2 + tol and numpy.abs(x[1]) > a/2 + tol:
            value[0] = omega*(sigma_x - sigma_y)/(omega**2 + sigma_y**2)
        
        elif numpy.abs(x[0]) > a/2 + tol and numpy.abs(x[1]) < a/2 + tol:
            value[0] = sigma_x/omega
            
        elif numpy.abs(x[0]) < a/2 + tol and numpy.abs(x[1]) > a/2 + tol:
            value[0] = -(omega*sigma_y)/(omega**2 + sigma_y**2)
            
        else:
            value[0] = 0
            
    def value_shape(self):
        return ()

            
class gR_xy(UserExpression):
    def eval(self, value, x):
        sigma_x = sigma*(numpy.abs(x[0])-a/2)**2
        sigma_y = sigma*(numpy.abs(x[1])-a/2)**2
        
        if numpy.abs(x[0]) > a/2 + tol and numpy.abs(x[1]) > a/2 + tol:
            value[0] = 1 - (sigma_x*sigma_y)/(omega**2)
        
        elif numpy.abs(x[0]) > a/2 + tol and numpy.abs(x[1]) < a/2 + tol:
            value[0] = 1
            
        elif numpy.abs(x[0]) < a/2 + tol and numpy.abs(x[1]) > a/2 + tol:
            value[0] = 1
            
        else:
            value[0] = 1
            
    def value_shape(self):
        return ()
            

class gI_xy(UserExpression):
    def eval(self, value, x):
        sigma_x = sigma*(numpy.abs(x[0])-a/2)**2
        sigma_y = sigma*(numpy.abs(x[1])-a/2)**2
        
        if numpy.abs(x[0]) > a/2 + tol and numpy.abs(x[1]) > a/2 + tol:
            value[0] = (sigma_y + sigma_x)/(omega)
        
        elif numpy.abs(x[0]) > a/2 + tol and numpy.abs(x[1]) < a/2 + tol:
            value[0] = sigma_x/omega
            
        elif numpy.abs(x[0]) < a/2 + tol and numpy.abs(x[1]) > a/2 + tol:
            value[0] = sigma_y/omega
            
        else:
            value[0] = 0
            
    def value_shape(self):
        return ()


#========== SUBDOMAINS
#===================================================================================
    
class Source(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0) and between(x[0], (-1/1000, 1/1000)))
    
class OuterBoundaries(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


#========== MESH AND BOUNDARIES 
#===================================================================================

nx, ny = 250, 250
mesh = RectangleMesh(Point(-a/2-t, -a/2-t), Point(a/2+t, a/2+t), nx, ny)
boundaries = MeshFunction("size_t", mesh, 1)
boundaries.set_all(0)

Source().mark(boundaries, 1)
OuterBoundaries().mark(boundaries, 2)


#========== COMPLEX SPACE AND FUNCTIONS
#===================================================================================

VR = FunctionSpace(mesh, "CG", 1)
VI = FunctionSpace(mesh, "CG", 1)
W = BlockFunctionSpace([VR, VI], restrict=[None, None])

(pR, pI) = BlockTrialFunction(W)
(qR, qI) = BlockTestFunction(W)


#========== BOUNDARY CONDITIONS
#===================================================================================

one = Constant(1.0)
BC1 = DirichletBC(W.sub(0), Constant(1.0), boundaries, 1)
BC2 = DirichletBC(W.sub(1), Constant(1.0), boundaries, 1)
#BC3 = DirichletBC(W.sub(0), Constant(0.0), boundaries, 2)
#BC4 = DirichletBC(W.sub(1), Constant(0.0), boundaries, 2)
bcs = BlockDirichletBC([BC1, BC2])


#========== VARIATIONAL FORMULATION
#===================================================================================

yR = [gR_y_x(), gR_x_y(), gR_xy()]
yI = [gI_y_x(), gI_x_y(), gI_xy()]
#yR = [1, 1, 1]
#yI = [1, 1, 1]

aF_11 =    yR[0]*pR.dx(0)*qR.dx(0) + yR[1]*pR.dx(1)*qR.dx(1) - yR[2]*kappa**2*inner(pR, qR)
aF_11 +=   yI[0]*pR.dx(0)*qR.dx(0) + yI[1]*pR.dx(1)*qR.dx(1) - yI[2]*kappa**2*inner(pR, qR)
aF_11 *=   dx

aF_22 =    yR[0]*pI.dx(0)*qI.dx(0) + yR[1]*pI.dx(1)*qI.dx(1) - yR[2]*kappa**2*inner(pI, qI)
aF_22 +=   yI[0]*pI.dx(0)*qI.dx(0) + yI[1]*pI.dx(1)*qI.dx(1) - yI[2]*kappa**2*inner(pI, qI)
aF_22 *=   dx

aF_12 =    yI[0]*pR.dx(0)*qI.dx(0) + yI[1]*pR.dx(1)*qI.dx(1) - yI[2]*kappa**2*inner(pR, qI)
aF_12 += - yR[0]*pR.dx(0)*qI.dx(0) - yR[1]*pR.dx(1)*qI.dx(1) + yR[2]*kappa**2*inner(pR, qI)
aF_12 *=   dx

aF_21 =  - yI[0]*pI.dx(0)*qR.dx(0) - yI[1]*pI.dx(1)*qR.dx(1) + yI[2]*kappa**2*inner(pI, qR)
aF_21 +=   yR[0]*pI.dx(0)*qR.dx(0) + yR[1]*pI.dx(1)*qR.dx(1) - yR[2]*kappa**2*inner(pI, qR)
aF_21 *=   dx

aF = [[aF_11, aF_12],
      [aF_21, aF_22]]

LF = [0, 0]
LF = BlockForm(LF, block_function_space=W, block_form_rank=1)

A = block_assemble(aF)
F = block_assemble(LF)
bcs.apply(A)
bcs.apply(F)


#========== SOLVE AND SAVE
#===================================================================================

sol = BlockFunction(W)
block_solve(A, sol.block_vector(), F, "mumps")
pR, pI = block_split(sol)
plt.figure()
plot(pR)
plt.figure()
plot(pI)
plt.show()

file_pR = File("Acoustic/pR.pvd")
file_pI = File("Acoustic/pI.pvd")
file_pR << pR
file_pI << pI
