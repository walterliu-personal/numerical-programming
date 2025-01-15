"""
Contains helpful classes for finite element calculations to be used in specific problems.
"""

import math
import numpy as np
from pyqtree import Index

from maths import *


def point_in_triangle(p, p0, p1, p2):
    """
    This only works in 2D space.
    """
    s = (p0.X - p2.X) * (p.Y - p2.Y) - (p0.Y - p2.Y) * (p.X - p2.X)
    t = (p1.X - p0.X) * (p.Y - p0.Y) - (p1.Y - p0.Y) * (p.X - p0.X)
    if ((s < 0) != (t < 0) and s != 0 and t != 0):
        return False
    d = (p2.X - p1.X) * (p.Y - p1.Y) - (p2.Y - p1.Y) * (p.X - p1.X)
    return (d == 0) or (d < 0) == (s + t <= 0)

class Edge(Line):

    def __init__(self, p1, p2, boundary=None):
        Line.__init__(self, p1, p2)
        self.boundary = boundary

    def __eq__(self, other):
        return (self.p1 == other.p1 and self.p2 == other.p2) or (self.p1 == other.p2 and self.p2 == other.p1)
    
    def __str__(self):
        print(self.p1, end="  ")
        print(self.p2, end="  ")
        return ""
    
    def __repr__(self):
        return repr(self.p1) + "  " + repr(self.p2)
    
    def __hash__(self):
        return int(self.p1.x * (10 ** 40)) + int(self.p1.y * (10 ** 30)) + int(self.p2.x * (10 ** 20)) + int(self.p2.y * (10 ** 10))

class Triangle2D:
    """
    Generic 2D triangle.
    """
    def __init__(self, p0, p1, p2):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

        # Derived quantities.
        self.area = 0.5 * abs(p0.X * (p1.Y - p2.Y) + p1.X * (p2.Y - p0.Y) + p2.X * (p0.Y - p1.Y))
        # Note convention.
        self.l01 = distance(p0, p1)
        self.l12 = distance(p1, p2)
        self.l20 = distance(p2, p0)

    def contains(self, point):
        return point_in_triangle(point, self.p0, self.p1, self.p2)
    
    def bbox(self):
        # Returns the bounding box of the triangle.
        minx = min([self.p0.x, self.p1.x, self.p2.x])
        miny = min([self.p0.y, self.p1.y, self.p2.y])
        maxx = max([self.p0.x, self.p1.x, self.p2.x])
        maxy = max([self.p0.y, self.p1.y, self.p2.y])
        return (minx, miny, maxx, maxy)
    
    def __str__(self):
        print(self.p0, end="  ")
        print(self.p1, end="  ")
        print(self.p2, end="  ")
        return ""

class TriangularElement2D(Triangle2D):
    """
    Generic 2D triangle element. Points should be ordered counterclockwise.
    """
    def __init__(self, p0, p1, p2):
        Triangle2D.__init__(self, p0, p1, p2)
        # Basis functions.
        self.phi0 = TriangleBasis2D(p0, p1, p2)
        self.phi1 = TriangleBasis2D(p1, p2, p0)
        self.phi2 = TriangleBasis2D(p2, p0, p1)


class TriangleBasis2D:
    """
    A 'triangular' linear function matching a value of 1 at one point
    and 0 at two others. Equals 0 outside of the triangle defined by the above three points.
    """
    def __init__(self, pole, p1, p2):
        # Function = 1 at the pole.
        # Convention: Points are defined in counterclockwise order, this class will detect and correct for this if needed.
        self.pole = pole
        self.p1 = p1
        self.p2 = p2
        # The base triangle.
        self.triangle = Triangle2D(self.pole, self.p1, self.p2)

        # Derived quantities.
        self.area = 0.5 * abs(pole.X * (p1.Y - p2.Y) + p1.X * (p2.Y - pole.Y) + p2.X * (pole.Y - p1.Y))
        # Function is of the form f(x, y) = a + bx + cy.
        self.a = (p1.X * p2.Y - p1.Y * p2.X) / (2 * self.area)
        self.b = (p1.Y - p2.Y) / (2 * self.area)
        self.c = (p2.X - p1.X) / (2 * self.area)
        # If f(x0, y0) = -1, triangle is defined clockwise and the signs of the coefficients need to be flipped.
        if np.isclose(self.a + self.b * pole.x + self.c * pole.y, -1):
            self.a = -self.a
            self.b = -self.b
            self.c = -self.c

    def __call__(self, x, y):
        if not point_in_triangle(Point(x, y), self.pole, self.p1, self.p2):
            return 0
        return self.a + self.b * x + self.c * y
    
    def __str__(self):
        return f"{self.a} + {self.b}x + {self.c}y"


class FEMSolution2D:
    """
    Generic finite element solution in 2D.
    Uses triangular meshes and a quadtree to efficiently compute each point.
    """
    def __init__(self, point_list, element_list, coefficients):
        """
        Point list: length = no. of nodes, containing Point objects.
        Element list: length = no. of elements, containing Triangle objects.
        Coefficients: length = no. of nodes
        """
        self.nodes = point_list
        self.elements = element_list
        self.coefficients = coefficients

        # Construct a quadtree to efficiently determine which element a given point is in.
        maxx = -float("inf")
        maxy = -float("inf")
        minx = float("inf")
        miny = float("inf")
        for node in self.nodes:
            if node.x < minx:
                minx = node.x
            if node.y < miny:
                miny = node.y
            if node.x > maxx:
                maxx = node.x
            if node.y > maxy:
                maxy = node.y
        self.quadtree = Index(bbox=(minx, miny, maxx, maxy))
        for element in self.elements:
            self.quadtree.insert(element, element.bbox())


    def __call__(self, x, y):
        matches = self.quadtree.intersect((x, y, x, y))
        point = Point(x, y)
        for element in matches:
            if element.contains(point):
                # If the point is a node:
                if element.p0 == point: return self.coefficients[element.p0.id]
                if element.p1 == point: return self.coefficients[element.p1.id]
                if element.p2 == point: return self.coefficients[element.p2.id]

                # Since the final function should be continuous, no need to take averages on element borders.
                return element.phi0(x, y) * self.coefficients[element.p0.id] + element.phi1(x, y) * self.coefficients[element.p1.id] + element.phi2(x, y) * self.coefficients[element.p2.id]
                
        

if __name__ == "__main__":
    # Testing area
    point_list = [Point(0, 0, id=0), Point(0, 1, id=1), Point(1, 1, id=2), Point(1, 0, id=3)]
    element_list = [TriangularElement2D(point_list[0], point_list[1], point_list[3]), TriangularElement2D(point_list[1], point_list[2], point_list[3])]
    coefficients = [1, 0.25, 0.5, 0.75]
    solution = FEMSolution2D(point_list, element_list, coefficients)
    print(solution(0.3, 0.9))
    print(solution(0.99, 0.99))
    print(solution(0.51, 0.51))
    print(solution(1, 0))