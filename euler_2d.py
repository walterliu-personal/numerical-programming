"""
Solves for potential flow in two dimensions by using continuity and the Navier-Stokes equations,
grossly simplified.
The flow is incompressible, inviscid (-> irrotational) and steady.
"""

"""
The governing equations are:
div(v) = 0                (continuity)
ρv • grad(v) = -grad(p)   (momentum balance (Navier-Stokes))
Since the flow is irrotational, we can write v = -grad(ϕ).
Then by continuity, -div(grad(ϕ)) = 0, i.e. Laplace's equation.
Finally, by Bernoulli's equation, p + 1/2 v^2 = const.
Hence the Bernoulli pressure = -(v • v) = -(grad(ϕ) • grad(ϕ)).
Imposing pressure boundary conditions will allow offseting to get the absolute pressure.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy as dcp
import random
from scipy import stats

import maths
from maths import *
from fem_base import *
from mesher import *

class PotentialFlowSolver2D:

    def __init__(self, bbox, wallfunc, V_in, resolution=5, tolerance=maths.tol):
        # The following BCs are used:
        # Inlet: -n • grad(ϕ) = V_in
        # Opening: ϕ = 0
        # Wall: -n • grad(ϕ) = 0
        # It is assumed that the left boundary is the inlet, other boundaries are openings, 
        # and a no-slip wall is applied to the object with boundaries defined by wallfunc.

        # Generate initial mesh
        self.mesh2D = FluidMesh2D(bbox, wallfunc, resolution)
        self.V_in = V_in
        self.bbox = bbox
        self.tolerance = tolerance


    def solve(self):
        start = time.time()
        # Form (n+1) x (n+1) stiffness matrix, A
        A = [[0 for j in range(self.mesh2D.num_nodes)] for i in range(self.mesh2D.num_nodes)]
        for element in self.mesh2D.elements:
            A[element.p0.id][element.p0.id] += element.area * (element.phi0.b ** 2 + element.phi0.c ** 2)
            A[element.p0.id][element.p1.id] += element.area * (element.phi0.b * element.phi1.b + element.phi0.c * element.phi1.c)
            A[element.p0.id][element.p2.id] += element.area * (element.phi0.b * element.phi2.b + element.phi0.c * element.phi2.c)
            A[element.p1.id][element.p0.id] += element.area * (element.phi1.b * element.phi0.b + element.phi1.c * element.phi0.c)
            A[element.p1.id][element.p1.id] += element.area * (element.phi1.b ** 2 + element.phi1.c ** 2)
            A[element.p1.id][element.p2.id] += element.area * (element.phi1.b * element.phi2.b + element.phi1.c * element.phi2.c)
            A[element.p2.id][element.p0.id] += element.area * (element.phi2.b * element.phi0.b + element.phi2.c * element.phi0.c)
            A[element.p2.id][element.p1.id] += element.area * (element.phi2.b * element.phi1.b + element.phi2.c * element.phi1.c)
            A[element.p2.id][element.p2.id] += element.area * (element.phi2.b ** 2 + element.phi2.c ** 2)
        A = np.array(A)

        # Form (n+1) x (n+1) residual matrix, R
        R = [[0 for j in range(self.mesh2D.num_nodes)] for i in range(self.mesh2D.num_nodes)]
        for edge in self.mesh2D.boundary_edges:
            if (edge.boundary == OPENING):
                R[edge.p1.id][edge.p1.id] += round((edge.length / 6) * (2/self.tolerance), -5)
                R[edge.p1.id][edge.p2.id] += round((edge.length / 6) * (1/self.tolerance), -5)
                R[edge.p2.id][edge.p1.id] += round((edge.length / 6) * (1/self.tolerance), -5)
                R[edge.p2.id][edge.p2.id] += round((edge.length / 6) * (2/self.tolerance), -5)
        R = np.array(R)

        # Form (n+1) residual vector, r
        # Note vector b = int(f * v dK) = 0, since f = 0 (special case of Poissons')
        r = [0 for i in range(self.mesh2D.num_nodes)]
        for edge in self.mesh2D.boundary_edges:
            if edge.boundary == INLET:
                r[edge.p1.id] += (edge.length / 2) * self.V_in
                r[edge.p2.id] += (edge.length / 2) * self.V_in
            elif edge.boundary == OUTLET:
                r[edge.p1.id] += (edge.length / 2) * -self.V_in
                r[edge.p2.id] += (edge.length / 2) * -self.V_in
        r = np.array(r)
        
        # Symmetry check
        try:
            assert np.allclose((A+R), (A+R).T)
        except:
            raise Warning("Stiffness matrix is not symmetric??")

        # Solve the linear system
        self.coeffs = np.linalg.solve(A+R, r)
        self.solution = FEMSolution2D(self.mesh2D.points, self.mesh2D.elements, self.coeffs)
        
        x, y, z = [], [], []
        real = []
        for p in self.mesh2D.points:
            x.append(p.x)
            y.append(p.y)
            #z.append(self.solution(p.x, p.y))
            r = float((p.x * p.x + p.y * p.y) ** 0.5)
            try:
                if p.x > 0:
                    theta = np.arctan(float(p.y) / float(p.x))
                elif p.x < 0 and p.y >= 0:
                    theta = np.arctan(float(p.y) / float(p.x)) + np.pi
                elif p.x < 0 and p.y < 0:
                    theta = np.arctan(float(p.y) / float(p.x)) - np.pi

            except ZeroDivisionError:
                if p.y > 0: theta = np.pi / 2
                else: p.y = -np.pi / 2
            real.append(-self.V_in * r * (1 + 4 / (r ** 2)) * np.cos(theta))
        print("Linear solution completed in", time.time() - start, "seconds.")
        print("Velocity: ", self.V_in)
        fig, (ax1, ax2) = plt.subplots(2)
        c1 = ax1.tricontourf(x, y, [[e.p0.id, e.p1.id, e.p2.id] for e in self.mesh2D.elements], self.coeffs, 40)
        c2 = ax2.tricontourf(x, y, [[e.p0.id, e.p1.id, e.p2.id] for e in self.mesh2D.elements], real, 40)
        plt.colorbar(c1, ax=ax1)
        plt.colorbar(c2, ax=ax2)
        plt.show()
        return self.solution
            
        
        


# Testing area       
if __name__ == "__main__":
    def rectangle(x, y):
        if (-3 < x < 3) and (-1 < y < 1):
            return False
        return True
    
    def cylinder(x, y):
        return x ** 2 + y ** 2 >= 2
    solution = PotentialFlowSolver2D((-10, -10, 10, 10), cylinder, 10, 80, 10 ** -8)
    solution.solve()