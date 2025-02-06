"""
Generates a uniform square triangular mesh which can be self-similarly refined.
"""

"""
E.g.
- Initial points (0, 0), (0, 1), (1, 1), (1, 0) with triangles (0, 1, 3), (1, 2, 3)
- Final points (0, 0), (0, 1), (1, 1), (1, 0), (0, 0.5), (0.5, 0), (0.5, 0.5), (0.5, 1), (1, 0.5)
  with triangles (0, 4, 5), (4, 5, 6), (5, 6, 3), (4, 6, 1), (1, 7, 6), (7, 6, 8), (7, 8, 2), (6, 8, 3)

Visualise using the following code:

import matplotlib.pyplot as plt
p = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0.5), (0.5, 0), (0.5, 0.5), (0.5, 1), (1, 0.5)]
plt.triplot([_[0] for _ in p], [_[1] for _ in p], [(0, 4, 5), (4, 5, 6), (5, 6, 3), (4, 6, 1), (1, 7, 6), (7, 6, 8), (7, 8, 2), (6, 8, 3)])
plt.show()

Each element is divided into 4 congurent triangles, preserving the quality of the triangles and allowing for a pretty easy algorithm.
Time complexity: quadratic with resolution, linear with elements.
"""

from copy import deepcopy as dcp
import matplotlib.pyplot as plt
import time
from scipy import stats

from fem_base import *

# Boundaries
INLET = 0 # -n • grad(ϕ) = V_in
OPENING = 1 # ϕ = 0 (velocity constant)
WALL = 2 # -n • grad(ϕ) = 0
OUTLET = 3 # -n • grad(ϕ) = -V_in


class SquareElement2D:
    # Not the base element, but this allows keeping track of the neighbour of a triangle element.
    def __init__(self, element1, element2):
        self.element1 = element1
        self.element2 = element2

class FluidMesh2D:

    def addpoint(self, point, pdict):
        try:
            return pdict[point.x][point.y]
            # print("A collision has happened!!")
        except:
            point.setid(self.num_nodes)
            self.points.append(point)
            try:
                pdict[point.x][point.y] = point.id
            except:
                pdict[point.x] = dict()
                pdict[point.x][point.y] = point.id
            self.num_nodes += 1

    def addelement(self, element):
        element.setid(self.num_elements)
        self.elements.append(element)
        element.p0.addelement(element.id)
        element.p1.addelement(element.id)
        element.p2.addelement(element.id)
        self.num_elements += 1

    def add_element_in_fluid(self, p0, p1, p2):
        wallp0 = self.wallfunc(p0.x, p0.y)
        wallp1 = self.wallfunc(p1.x, p1.y)
        wallp2 = self.wallfunc(p2.x, p2.y)

        trig_element = 0

    def add_square_element_in_fluid(self, ld, rd, rt, lt):
        try:
            wallld = self.wallfunc(ld.x, ld.y)
        except:
            wallld = None
        try:
            walllt = self.wallfunc(lt.x, lt.y)
        except:
            walllt = None
        try:
            wallrd = self.wallfunc(rd.x, rd.y)
        except:
            wallrd = None
        try:
            wallrt = self.wallfunc(rt.x, rt.y)
        except:
            wallrt = None
        addld, addlt, addrd, addrt = False, False, False, False
        
        if (wallld and walllt and wallrt and (not wallrd)):
            
            trig_element1 = TriangularElement2D(ld, rt, lt, orientation=SE)
            self.addelement(trig_element1)
            self.boundary_edges.add(Edge(ld, rt, WALL))
            addld, addrt, addlt = True, True, True
        
        elif (walllt and wallrt and wallrd and (not wallld)):
            
            trig_element1 = TriangularElement2D(lt, rd, rt, orientation=SW)
            self.addelement(trig_element1)
            self.boundary_edges.add(Edge(lt, rd, WALL))
            addlt, addrd, addrt = True, True, True

        elif (wallld and walllt and wallrd and (not wallrt)):
            
            trig_element1 = TriangularElement2D(ld, rd, lt, orientation=NE)
            self.addelement(trig_element1)
            self.boundary_edges.add(Edge(lt, rd, WALL))
            addld, addrd, addlt = True, True, True

        elif (wallld and wallrd and wallrt and (not walllt)):
            
            trig_element1 = TriangularElement2D(ld, rd, rt, orientation=NW)
            self.addelement(trig_element1)
            self.boundary_edges.add(Edge(ld, rt, WALL))
            addld, addrd, addrt = True, True, True

        elif (wallld and wallrd and (not walllt and not wallrt)):
            self.boundary_edges.add(Edge(ld, rd, WALL))
            
        elif (wallld and walllt and (not wallrd and not wallrt)):
            self.boundary_edges.add(Edge(ld, lt, WALL))
            
        elif (wallrd and wallrt and (not walllt and not wallld)):
            self.boundary_edges.add(Edge(rd, rt, WALL))
            
        elif (walllt and wallrt and (not wallld and not wallrd)):
            self.boundary_edges.add(Edge(lt, rt, WALL))
            
        elif (wallld and walllt and wallrd and wallrt):
            trig_element1 = TriangularElement2D(ld, rd, lt, orientation=NE)
            self.addelement(trig_element1)
            trig_element2 = TriangularElement2D(lt, rd, rt, orientation=SW)
            self.addelement(trig_element2)
            
            addld, addrd, addlt, addrt = True, True, True, True
            
        if addld:
            try:
                self.points_dict[ld.x][ld.y]
            except:
                self.addpoint(ld, self.points_dict)
        if addrd:
            try:
                self.points_dict[rd.x][rd.y]
            except:
                self.addpoint(rd, self.points_dict)
        if addlt:
            try:
                self.points_dict[lt.x][lt.y]
            except:
                self.addpoint(lt, self.points_dict)
        if addrt:
            try:
                self.points_dict[rt.x][rt.y]
            except:
                self.addpoint(rt, self.points_dict)
                
    
    def __init__(self, bbox, wallfunc, initial_resolution=5):
        """
        wallfunc(x, y) = True if in fluid, False if in solid.
        Values right on boundary = True.
        It is assumed that the left boundary is the inlet, other boundaries are openings, 
        and a no-slip wall is applied to the object with boundaries defined by wallfunc.
        """
        start = time.time()

        self.xlen = bbox[2] - bbox[0]
        self.ylen = bbox[3] - bbox[1]
        resolution = Fraction(math.gcd(self.xlen, self.ylen), initial_resolution)
        self.x_spacing = int(round(self.xlen / resolution))
        self.y_spacing = int(round(self.ylen / resolution))
        self.points = []
        self.boundary_edges = set()
        self.elements = []
        self.square_elements = []
        self.points_dict = dict()
        self.wallfunc = wallfunc

        self.num_nodes = 0
        self.num_elements = 0
        for step_y in range(int(round(self.ylen / resolution))):
            for step_x in range(int(round(self.xlen / resolution))):
                newx = bbox[0] + step_x * resolution
                newy = bbox[1] + step_y * resolution
                # Check if any of the bounding box corners are in a fluid.
                if (wallfunc(newx, newy) or wallfunc(newx + resolution, newy) or wallfunc(newx, newy + resolution) or wallfunc(newx + resolution, newy + resolution)):
                    try:
                        ld = self.points[self.points_dict[newx][newy]]
                    except:
                        ld = Node(newx, newy)
                    try:
                        lt = self.points[self.points_dict[newx][newy + resolution]]
                    except:
                        lt = Node(newx, newy + resolution)
                    try:
                        rd = self.points[self.points_dict[newx + resolution][newy]]
                    except:
                        rd = Node(newx + resolution, newy)
                    try:
                        rt = self.points[self.points_dict[newx + resolution][newy + resolution]]
                    except:
                        rt = Node(newx + resolution, newy + resolution)
                    
                    self.add_square_element_in_fluid(ld, rd, rt, lt)
                        
        print(len(self.boundary_edges))
        # Add boundary edges
        for step_y in range(int(round(self.ylen / resolution)) + 1):
            x1 = bbox[0]
            x2 = bbox[2]
            y = bbox[1] + step_y * resolution
            try:
                current_point_1 = self.points[self.points_dict[x1][y]]
                try:
                    point_up = self.points[self.points_dict[x1][y + resolution]]
                    self.boundary_edges.add(Edge(current_point_1, point_up, INLET))
                except:
                    pass
            except:
                pass
            try:
                current_point_2 = self.points[self.points_dict[x2][y]]
                try:
                    point_up = self.points[self.points_dict[x2][y + resolution]]
                    self.boundary_edges.add(Edge(current_point_2, point_up, OUTLET))
                except:
                    pass
            except:
                pass
        for step_x in range(int(round(self.xlen / resolution)) + 1):
            y1 = bbox[1]
            y2 = bbox[3]
            x = bbox[0] + step_x * resolution
            try:
                current_point_1 = self.points[self.points_dict[x][y1]]
                try:
                    point_to_right = self.points[self.points_dict[x + resolution][y1]]
                    self.boundary_edges.add(Edge(current_point_1, point_to_right, WALL))
                except:
                    pass
            except:
                pass
            try:
                current_point_2 = self.points[self.points_dict[x][y2]]
                try:
                    point_to_right = self.points[self.points_dict[x + resolution][y2]]
                    self.boundary_edges.add(Edge(current_point_2, point_to_right, WALL))
                except:
                    pass
            except:
                pass


        print("Initial mesh statistics:")
        print("Number of nodes: ", self.num_nodes, len(self.points))
        print("Number of elements: ", self.num_elements)
        print(len(self.boundary_edges))
        self.t_used = time.time() - start
        print("Time used: ", self.t_used, "seconds.")
        plt.triplot([p.x for p in self.points], [p.y for p in self.points], [[e.p0.id, e.p1.id, e.p2.id] for e in self.elements])
        plt.show()

    def refine(self, solver, threshold=0.2, visits=1):
        # Refines the mesh based on the element residual defined by Larson and Bengzon, 4.143:
        # μ_K = h_K ||f + div(grad(u_h))|| + (1/2) * ((h_K) ** (1/2)) * lineint([n • grad(u_h)] d∂K)
        # Where [n • grad(u_h)] is the jump of the piecewise constant gradient of the FE approximation.
        solution = solver.coeffs
        running_mean = 0
        M2 = 0
        readings = 0

        print("Initial number of elements: ", self.num_elements)
        print("Refining mesh...")
        refined_elements = []
        start = time.time()
        while readings < int(visits * self.num_elements):
            index = random.randint(0, self.num_elements - 1)
            current_element = self.elements[index]
            print(current_element.id)
            if current_element.id not in refined_elements:
                neighbours = set()
                neighbours_and_edge = []
                edges = []

                # Find the neighbours of this element.
                for el0 in current_element.p0.elements:
                    if el0 != current_element.id:
                        if el0 in current_element.p1.elements:
                            a = len(neighbours)
                            neighbours.add(el0)
                            if len(neighbours) != a:
                                neighbours_and_edge.append((el0, current_element.l01))
                                edges.append(current_element.l01)
                        if el0 in current_element.p2.elements:
                            a = len(neighbours)
                            neighbours.add(el0)
                            if len(neighbours) != a:
                                neighbours_and_edge.append((el0, current_element.l20))
                                edges.append(current_element.l20)
                for el1 in current_element.p1.elements:
                    if el1 != current_element.id:
                        if el1 in current_element.p2.elements:
                            a = len(neighbours)
                            neighbours.add(el1)
                            if len(neighbours) != a:
                                neighbours_and_edge.append((el1, current_element.l12))
                                edges.append(current_element.l12)
                if current_element.l01 not in edges: edges.append(current_element.l01)
                if current_element.l12 not in edges: edges.append(current_element.l12)
                if current_element.l20 not in edges: edges.append(current_element.l20)

                readings += 1
                # Calculate the element resiudal.
                h_K = max(edges[0], edges[1], edges[2], key=lambda x: x.length).length
                residual = 0
                dx0 = current_element.phi0.b * solution[current_element.p0.id] + current_element.phi1.b * solution[current_element.p1.id] + current_element.phi2.b * solution[current_element.p2.id]
                dy0 = current_element.phi0.c * solution[current_element.p0.id] + current_element.phi1.c * solution[current_element.p1.id] + current_element.phi2.c * solution[current_element.p2.id]
                for (neighbour, edge) in neighbours_and_edge:
                    dx1 = self.elements[neighbour].phi0.b * solution[self.elements[neighbour].p0.id] + self.elements[neighbour].phi1.b * solution[self.elements[neighbour].p1.id] + self.elements[neighbour].phi2.b * solution[self.elements[neighbour].p2.id]
                    dy1 = self.elements[neighbour].phi0.c * solution[self.elements[neighbour].p0.id] + self.elements[neighbour].phi1.c * solution[self.elements[neighbour].p1.id] + self.elements[neighbour].phi2.c * solution[self.elements[neighbour].p2.id]
                    lx = edge.p2.x - edge.p1.x
                    ly = edge.p2.y - edge.p1.y
                    nx_plus = -ly
                    ny_plus = lx
                    nx_minus = ly
                    ny_minus = -lx
                    jump = (nx_plus * dx0 + ny_plus * dy0) + (nx_minus * dx1 + ny_minus * dy1)
                    residual += jump
                residual = (1/2) * (h_K ** 0.5) * residual
                residual += solver.poisson_func(current_element.p0.x, current_element.p0.y) * current_element.area * h_K / 3
                residual += solver.poisson_func(current_element.p1.x, current_element.p1.y) * current_element.area * h_K / 3
                residual += solver.poisson_func(current_element.p2.x, current_element.p2.y) * current_element.area * h_K / 3

                # Check if residual is unusally high
                delta = residual - running_mean
                running_mean += delta / readings
                delta2 = residual - running_mean
                M2 += (delta * delta2)
                variance = M2 / readings
                prob = stats.norm.cdf(residual, running_mean, variance ** (1/2))
                if (prob < threshold) or (prob > 1 - threshold):
                    # Refine element.
                    if current_element.element_type == TYPE45:
                        refined_elements.append(current_element.id)

                        # Create 4 new self-similar elements.
                        try:
                            new_point_01 = self.points_dict[Fraction(1, 2) * (current_element.p0.x + current_element.p1.x)][Fraction(1, 2) * (current_element.p0.y + current_element.p1.y)]
                        except:
                            new_point_01 = mid(current_element.p0, current_element.p1)
                        try:
                            new_point_12 = self.points_dict[Fraction(1, 2) * (current_element.p1.x + current_element.p2.x)][Fraction(1, 2) * (current_element.p1.y + current_element.p2.y)]
                        except:
                            new_point_12 = mid(current_element.p1, current_element.p2)
                        try:
                            new_point_20 = self.points_dict[Fraction(1, 2) * (current_element.p2.x + current_element.p0.x)][Fraction(1, 2) * (current_element.p2.y + current_element.p0.y)]
                        except:
                            new_point_20 = mid(current_element.p2, current_element.p0)

                        self.add_element_in_fluid()


    

# Testing area
if __name__ == "__main__":
    mesh = FluidMesh2D((-5, -5, 5, 5), lambda x, y: x ** 2 + y ** 2 > 3, 3)
    mesh.refine(1,1)

