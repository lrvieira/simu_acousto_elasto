from numpy import isclose
from dolfin import *
import matplotlib.pyplot as plt
from multiphenics import *


# ********* Model constants  ******* #
E, nu = Constant(1e5), Constant(0.)
mu = E/2./(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)
rho = Constant(1.2)
omega = 1
k = 15


# ******* Construct mesh and define normal, tangent ****** #
fluid = 10
solid = 11
left = 12
right = 13
interf = 14
wallF = 15
wallS = 16


# ******* Set subdomains, boundaries, and interface ****** #
mesh = RectangleMesh(Point(-2.0, -2.0), Point(2.0, 2.0), 30, 30)
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)
boundaries = MeshFunction("size_t", mesh, 1)
boundaries.set_all(0)

class TopF(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 2.0) and between(x[0], (-2.0, 0.0)) and on_boundary)
    
class TopS(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 2.0) and between(x[0], (0.0, 2.0)) and on_boundary)
    
class BottomF(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], -2.0) and between(x[0], (-2.0, 0.0)) and on_boundary)
    
class BottomS(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], -2.0) and between(x[0], (0.0, 2.0)) and on_boundary)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], -2.0) and on_boundary)
    
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 2.0) and on_boundary)

class Fluid(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= 0.

class Solid(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= 0.

class Interface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

Fluid().mark(subdomains, fluid)
Solid().mark(subdomains, solid)
Interface().mark(boundaries, interf)

TopF().mark(boundaries, wallF)
TopS().mark(boundaries, wallS)
BottomF().mark(boundaries, wallF)
BottomS().mark(boundaries, wallS)
Left().mark(boundaries, left)
Right().mark(boundaries, right)


# ******* Set subdomains, boundaries, and interface ****** #
OmF = MeshRestriction(mesh, Fluid())
OmS = MeshRestriction(mesh, Solid())
Gam = MeshRestriction(mesh, Interface()) #Is it needed?

dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
dS = Measure("dS", domain=mesh, subdomain_data=boundaries)

sigma = 2.0*mu*sym(grad(u)) + lmbda*tr(sym(grad(u)))*Identity(u.geometric_dimension())
n = FacetNormal(mesh)


# verifying the domain size
areaF = assemble(1.*dx(fluid))
areaS = assemble(1.*dx(solid))
lengthI = assemble(1.*dS(interf))
print("area(Omega_D) = ", areaF)
print("area(Omega_S) = ", areaS)
print("length(Gamma) = ", lengthI)


# ***** Global FE spaces and their restrictions ****** #
P2v = VectorFunctionSpace(mesh, "CG", 2)
P1 = FunctionSpace(mesh, "CG", 1)

W = BlockFunctionSpace([P1, P2v], restrict=[OmF, OmS])

trial = BlockTrialFunction(W)
test = BlockTestFunction(W)
p, u = block_split(trial)
q, v = block_split(test)

#'Pt', 'la' and 'xi' concern the interface, right?

print("DoFs = ", W.dim())


# ******** Other parameters and BCs ************* #
source = Expression(("cos(k*x[0])"), degree=1, k=k)
noSlipS = Constant((0., 0.))
noSlipF = Constant(0.0)

bcPin = DirichletBC(W.sub(0), source, boundaries, left)
bcF = DirichletBC(W.sub(0), noSlipF, boundaries, wallF)
bcS = DirichletBC(W.sub(1), noSlipS, boundaries, wallS)
bcs = BlockDirichletBC([bcPin, bcF, bcS])


# ********  Define weak forms ********** #
a_11 = (inner(grad(p),grad(q)) - k**2 * p*q)*dx(fluid)
a_12 = (rho*omega**2 * avg(q) * inner(n('+'), u('+')))*dS(interf)
a_21 = (avg(p)*inner(n('+'), v('+')))*dS(interf)
a_22 = (inner(sigma, grad(v)) - rho*omega**2*inner(u,v))*dx(solid)

#Weak form
a = [[a_11, a_12],
     [a_21, a_22]]

L = [0, 0]

A = block_assemble(a)
F = block_assemble(L)
bcs.apply(A)
bcs.apply(F)

sol = BlockFunction(W)
block_solve(A, sol.block_vector(), F, "mumps")
