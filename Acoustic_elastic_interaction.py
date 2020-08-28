from numpy import isclose
#from dolfin import *
from fenics import *
import matplotlib.pyplot as plt
from multiphenics import *


# ********* Model constants  ******* #
E, nu = Constant(10e5), Constant(0.3)
mu = E/2./(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)
rho = Constant(1.2)
omega = 1
k = 15
x_1, y_1 = -4.0, -8.0
x_2, y_2 = 4.0, 8.0
nx, ny = 200, 400
int_x = 0.0
esp_x = 1.0


# ******* Construct mesh and define normal, tangent ****** #
fluid = 10
solid = 11
left = 12
right = 13
interf = 14
topF = 17
topS = 18
botF = 19
botS = 20
solL = 21
solR = 22


# ******* Set subdomains, boundaries, and interface ****** #
mesh = RectangleMesh(Point(x_1, y_1), Point(x_2, y_2), nx, ny)
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)
boundaries = MeshFunction("size_t", mesh, 1)
boundaries.set_all(0)

class TopF(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], y_2) and between(x[0], (x_1, int_x)) and between(x[0], ((int_x+esp_x), x_2)) and on_boundary)

class TopS(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], y_2) and between(x[0], (int_x, (int_x+esp_x))) and on_boundary)

class BottomF(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], y_1) and between(x[0], (x_1, int_x)) and between(x[0], ((int_x+esp_x), x_2)) and on_boundary)

class BottomS(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], y_1) and between(x[0], (int_x, (int_x+esp_x))) and on_boundary)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], x_1) and on_boundary)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], x_2) and on_boundary)

class Fluid(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] <= int_x or x[0] >= (int_x+esp_x))

class Solid(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] >= int_x and x[0] <= (int_x+esp_x))

class Interface(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], int_x) or near(x[0], (int_x+esp_x)))

Fluid().mark(subdomains, fluid)
Solid().mark(subdomains, solid)
Interface().mark(boundaries, interf)

TopF().mark(boundaries, topF)
TopS().mark(boundaries, topS)
BottomF().mark(boundaries, botF)
BottomS().mark(boundaries, botS)
Left().mark(boundaries, left)
Right().mark(boundaries, right)


#plt.figure()
#plot(mesh)
#plot(subdomains)
#plt.show()

# ******* Set subdomains, boundaries, and interface ****** #
OmF = MeshRestriction(mesh, Fluid())
OmS = MeshRestriction(mesh, Solid())
Gam = MeshRestriction(mesh, Interface()) #Is it needed?

dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
dS = Measure("dS", domain=mesh, subdomain_data=boundaries)

sigma = lambda u: 2.0*mu*sym(grad(u)) + lmbda*tr(sym(grad(u)))*Identity(u.geometric_dimension())
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

BC_Pin = DirichletBC(W.sub(0), 100, boundaries, left)
BC_botF = DirichletBC(W.sub(0), Constant(0.), boundaries, botF)
BC_topF = DirichletBC(W.sub(0), Constant(0.), boundaries, topF)
BC_botS = DirichletBC(W.sub(1), Constant((0., 0.)), boundaries, botS)
BC_topS = DirichletBC(W.sub(1), Constant((0., 0.)), boundaries, topS)
BC_interf = DirichletBC(W.sub(1), Constant((0., 0.)), boundaries, interf)
BC_left = DirichletBC(W.sub(0), Constant(0.), boundaries, topF)
BC_test = DirichletBC(W.sub(1).sub(1), Constant(0.), boundaries, botS)
bcs = BlockDirichletBC([BC_Pin, BC_topS, BC_botS])
#bcs = BlockDirichletBC([BC_test, BC_topS, BC_botS])


# ********  Define weak forms ********** #
a_11 = (inner(grad(p),grad(q)) - k**2 * p*q)*dx(fluid)
a_12 = (rho*omega**2 * avg(q) * inner(n('+'), u('+')))*dS(interf)
a_21 = (avg(p)*inner(n('+'), v('+')))*dS(interf)
a_22 = (inner(sigma(u), grad(v)) - rho*omega**2*inner(u,v))*dx(solid)

#Weak form
a = [[a_11, a_12],
     [a_21, a_22]]

L = [0, 0]
L = BlockForm(L, block_function_space=W, block_form_rank=1)  # only necessary because L is all zeros, otherwise multiphenics detects function spaces and rank automatically

A = block_assemble(a)
F = block_assemble(L)
bcs.apply(A)
bcs.apply(F)

#delta = PointSource(W.sub(0), Point(-1.0, 0.0), 10)
#delta.apply(F)

sol = BlockFunction(W)
block_solve(A, sol.block_vector(), F, "mumps")
p_sol, u_sol = block_split(sol)

file_p = File("solution_p.pvd")
file_u = File("solution_u.pvd")
file_p << p_sol
file_u << u_sol

plt.figure()
plot(p_sol)
plt.figure()
plot(u_sol)
plt.show()
