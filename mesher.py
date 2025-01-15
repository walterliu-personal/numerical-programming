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
from fractions import Fraction
import matplotlib.pyplot as plt
import time

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
            pdict[point.x][point.y]
            print("A collision has happened!!")
        except:
            point.setid(self.num_nodes)
            self.points.append(point)
            try:
                pdict[point.x][point.y] = point
            except:
                pdict[point.x] = dict()
                pdict[point.x][point.y] = point
            self.num_nodes = len(self.points)
    
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
        self.points_dict = {}

        self.num_nodes = 0
        self.num_elements = 0
        for step_y in range(int(round(self.ylen / resolution))):
            for step_x in range(int(round(self.xlen / resolution))):
                newx = bbox[0] + step_x * resolution
                newy = bbox[1] + step_y * resolution
                # Check if any of the bounding box corners are in a fluid.
                if (wallfunc(newx, newy) or wallfunc(newx + resolution, newy) or wallfunc(newx, newy + resolution) or wallfunc(newx + resolution, newy + resolution)):
                    try:
                        ld = self.points_dict[newx][newy]
                    except:
                        ld = Point(newx, newy)
                    try:
                        lt = self.points_dict[newx][newy + resolution]
                    except:
                        lt = Point(newx, newy + resolution)
                    try:
                        rd = self.points_dict[newx + resolution][newy]
                    except:
                        rd = Point(newx + resolution, newy)
                    try:
                        rt = self.points_dict[newx + resolution][newy + resolution]
                    except:
                        rt = Point(newx + resolution, newy + resolution)
                    
                    if ((lt - ld == Point(0, resolution) == rt - rd) and (rt - lt == Point(resolution, 0) == rd - ld)):
                        wallld = wallfunc(ld.x, ld.y)
                        walllt = wallfunc(lt.x, lt.y)
                        wallrd = wallfunc(rd.x, rd.y)
                        wallrt = wallfunc(rt.x, rt.y)
                        addld, addlt, addrd, addrt = False, False, False, False
                        square = True
                        if (wallld and walllt and wallrt and (not wallrd)):
                            
                            trig_element1 = TriangularElement2D(ld, rt, lt)
                            self.num_elements += 1
                            square = SquareElement2D(trig_element1, None)
                            self.elements.append(trig_element1)
                            self.boundary_edges.add(Edge(ld, rt, WALL))
                            addld, addrt, addlt = True, True, True
                        
                        elif (walllt and wallrt and wallrd and (not wallld)):
                            
                            trig_element1 = TriangularElement2D(lt, rd, rt)
                            self.num_elements += 1
                            square = SquareElement2D(trig_element1, None)
                            self.elements.append(trig_element1)
                            self.boundary_edges.add(Edge(lt, rd, WALL))
                            addlt, addrd, addrt = True, True, True

                        elif (wallld and walllt and wallrd and (not wallrt)):
                            
                            trig_element1 = TriangularElement2D(ld, rd, lt)
                            self.num_elements += 1
                            square = SquareElement2D(trig_element1, None)
                            self.elements.append(trig_element1)
                            self.boundary_edges.add(Edge(lt, rd, WALL))
                            addld, addrd, addlt = True, True, True

                        elif (wallld and wallrd and wallrt and (not walllt)):
                            
                            trig_element1 = TriangularElement2D(ld, rd, rt)
                            self.num_elements += 1
                            square = SquareElement2D(trig_element1, None)
                            self.elements.append(trig_element1)
                            self.boundary_edges.add(Edge(ld, rt, WALL))
                            addld, addrd, addrt = True, True, True

                        elif (wallld and wallrd and (not walllt and not wallrt)):
                            self.boundary_edges.add(Edge(ld, rd, WALL))
                            square = False

                        elif (wallld and walllt and (not wallrd and not wallrt)):
                            self.boundary_edges.add(Edge(ld, lt, WALL))
                            square = False

                        elif (wallrd and wallrt and (not walllt and not wallld)):
                            self.boundary_edges.add(Edge(rd, rt, WALL))
                            square = False
                            
                        elif (walllt and wallrt and (not wallld and not wallrd)):
                            self.boundary_edges.add(Edge(lt, rt, WALL))
                            square = False

                        elif (wallld and walllt and wallrd and wallrt):
                            trig_element1 = TriangularElement2D(ld, rd, lt)
                            trig_element2 = TriangularElement2D(lt, rd, rt)
                            self.num_elements += 2
                            square = SquareElement2D(trig_element1, trig_element2)
                            self.elements.append(trig_element1)
                            self.elements.append(trig_element2)
                            
                            addld, addrd, addlt, addrt = True, True, True, True

                        if square: self.square_elements.append(square)
                        if addld:
                            try:
                                self.points_dict[newx][newy]
                            except:
                                self.addpoint(ld, self.points_dict)
                        if addrd:
                            try:
                                self.points_dict[newx + resolution][newy]
                            except:
                                self.addpoint(rd, self.points_dict)
                        if addlt:
                            try:
                                self.points_dict[newx][newy + resolution]
                            except:
                                self.addpoint(lt, self.points_dict)
                        if addrt:
                            try:
                                self.points_dict[newx + resolution][newy + resolution]
                            except:
                                self.addpoint(rt, self.points_dict)
                        
        print(len(self.boundary_edges))
        # Add boundary edges
        for step_y in range(int(round(self.ylen / resolution)) + 1):
            x1 = bbox[0]
            x2 = bbox[2]
            y = bbox[1] + step_y * resolution
            try:
                current_point_1 = self.points_dict[x1][y]
                try:
                    point_up = self.points_dict[x1][y + resolution]
                    self.boundary_edges.add(Edge(current_point_1, point_up, INLET))
                except:
                    pass
            except:
                pass
            try:
                current_point_2 = self.points_dict[x2][y]
                try:
                    point_up = self.points_dict[x2][y + resolution]
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
                current_point_1 = self.points_dict[x][y1]
                try:
                    point_to_right = self.points_dict[x + resolution][y1]
                    self.boundary_edges.add(Edge(current_point_1, point_to_right, WALL))
                except:
                    pass
            except:
                pass
            try:
                current_point_2 = self.points_dict[x][y2]
                try:
                    point_to_right = self.points_dict[x + resolution][y2]
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
        plt.scatter([p.x for p in self.points], [p.y for p in self.points])
        plt.show()

# Testing area
if __name__ == "__main__":
    mesh = FluidMesh2D((-5, -5, 5, 5), lambda x, y: x ** 2 + y ** 2 > 3, 20)

