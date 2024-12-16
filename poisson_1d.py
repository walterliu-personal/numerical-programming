"""
Solves Poisson's equation in one dimension using the Finite Element Method
i.e. -u'' = f(x).
This is readily solved analytically for most simple functions f and is only implemented as an exercise.
Detects high gradients in the solution and applies adaptive mesh refinement accordingly.
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy as dcp
import random
from scipy import stats

from maths import *

def retconstfunc(const):
    return lambda x: const

class PiecewiseFunction:

    def __init__(self, seps):
        self.seps = seps

    def __call__(self, x):
        if not self.seps:
            return 0
        c = 0
        for interval, (funcleft, funcright) in self.seps:
            if c == len(self.seps) - 1:
                if x == interval:
                    return funcleft(x)
            if x < interval:
                return funcleft(x)
            c += 1
        return funcright(x)

class Hat:

    def __init__(self, i, h_i, h_j):
        self.i = i
        self.h_i = h_i
        self.h_j = h_j
        if h_i == 0 and h_j == 0:
            self.f = PiecewiseFunction([])
        elif h_i == 0:
            self.f = PiecewiseFunction([(i, (lambda x: 0, lambda x: 1)), (i + h_j, (lambda x: (i + h_j - x) / h_j, lambda x: 0))])
        elif h_j == 0:
            self.f = PiecewiseFunction([(i - h_i, (lambda x: 0, lambda x: 0)), (i, (lambda x: (x - (i - h_i)) / h_i, lambda x: 0))])
        else:
            self.f = PiecewiseFunction([(i - h_i, (lambda x: 0, lambda x: 0)), (i, (lambda x: (x - (i - h_i)) / h_i, lambda x: 1)), (i + h_j, (lambda x: (i + h_j - x) / h_j, lambda x: 0))])

    def __call__(self, x):
        return self.f(x)
    
    def diff(self):
        if self.h_i == 0 and self.h_j == 0:
            return PiecewiseFunction([])
        elif self.h_i == 0:
            return PiecewiseFunction([(self.i, (lambda x: 0, lambda x: -1 / self.h_j)), (self.i + self.h_j, (lambda x: -1 / self.h_j, lambda x: 0))])
        elif self.h_j == 0:
            return PiecewiseFunction([(self.i - self.h_i, (lambda x: 0, lambda x: 1 / self.h_i)), (self.i, (lambda x: 1 / self.h_i, lambda x: 0))])
        else:
            return PiecewiseFunction([(self.i - self.h_i, (lambda x: 0, lambda x: 1 / self.i)), (self.i, (lambda x: 1 / self.h_i, lambda x: -1 / self.h_j)), (self.i + self.h_j, (lambda x: -1 / self.h_j, lambda x: 0))])

class CompositeHat:
    '''
    A typical solution to a 1D Finite Element Problem.
    '''
    # Too lazy, could be a PiecewiseFunction

    def __init__(self, hatfuncs, coeffs, mesh1d, func):
        '''
        Everything needs to be of length n + 1, for consistency.
        '''
        self.hatfuncs = hatfuncs
        self.coeffs = coeffs
        self.mesh1d = mesh1d
        self.func = func

    def __call__(self, x):
        if x > self.mesh1d[-1]:
            return 0
        if x < self.mesh1d[0]:
            return 0
        for i in range(len(self.mesh1d) + 1):
            if i == 0:
                if self.mesh1d[i] <= x <= self.mesh1d[i + 1]:
                    return self.coeffs[i] * self.hatfuncs[i](x) + self.coeffs[i + 1] * self.hatfuncs[i + 1](x) 
            elif i == len(self.mesh1d):
                if self.mesh1d[i - 1] <= x <= self.mesh1d[i]:    
                    return self.coeffs[i - 1] * self.hatfuncs[i - 1](x) + self.coeffs[i] * self.hatfuncs[i](x)
            else:
                if self.mesh1d[i - 1] <= x <= self.mesh1d[i + 1]:    
                    return self.coeffs[i - 1] * self.hatfuncs[i - 1](x) + self.coeffs[i] * self.hatfuncs[i](x) + self.coeffs[i + 1] * self.hatfuncs[i + 1](x)

    
    def difflist(self):
        l = []
        for i in range(len(self.mesh1d)):
            if i == 0:
                l.append((-self.coeffs[0] + self.coeffs[1]) / (self.mesh1d[1] - self.mesh1d[0]))
            elif i == len(self.mesh1d) - 1:
                l.append((-self.coeffs[-2] + self.coeffs[-1]) / (self.mesh1d[-1] - self.mesh1d[-2]))
            else:
                wd1 = -self.coeffs[i - 1] / (self.mesh1d[i] - self.mesh1d[i - 1]) ** 2 + self.coeffs[i + 1] / (self.mesh1d[i + 1] - self.mesh1d[i]) ** 2
                wd2 = self.coeffs[i] / (self.mesh1d[i] - self.mesh1d[i - 1]) ** 2 - self.coeffs[i] / (self.mesh1d[i + 1] - self.mesh1d[i]) ** 2
                norm = (1 / (self.mesh1d[i + 1] - self.mesh1d[i])) + (1 / (self.mesh1d[i] - self.mesh1d[i - 1]))
                l.append((wd1 + wd2) / norm)
        return l
    
    def second_difflist(self):
        dl = self.difflist()
        # Quadratic interpolation approximation of the second derivative from exact values of the first derivative, calculated in a different function.
        # Somehow, this is equal to using a weighted average of the forward and backward difference (only proved empirically!)
        ddl = []
        for i in range(len(dl)):
            if i == 0:
                # First order approximation only.
                ddl.append((dl[1] - dl[0]) / (self.mesh1d[1] - self.mesh1d[0]))
            elif i == len(dl) - 1:
                ddl.append((dl[-1] - dl[-2]) / (self.mesh1d[-1] - self.mesh1d[-2]))
            else:
                wd1 = (dl[i + 1] - dl[i]) / (self.mesh1d[i + 1] - self.mesh1d[i]) ** 2
                wd2 = (dl[i] - dl[i - 1]) / (self.mesh1d[i - 1] - self.mesh1d[i]) ** 2
                norm = (1 / (self.mesh1d[i + 1] - self.mesh1d[i])) + (1 / (self.mesh1d[i] - self.mesh1d[i - 1]))
                ddl.append((wd1 + wd2) / norm)
        return ddl
         

    def diff(self):
        seps = [(self.mesh1d[0], (lambda x: 0, retconstfunc((self.coeffs[0] - self.coeffs[1]) / (self.mesh1d[1] - self.mesh1d[0]))))]
        for i in range(1, len(self.mesh1d) - 1):
            seps.append((self.mesh1d[i], (retconstfunc((-self.coeffs[i - 1] + self.coeffs[i]) / (self.mesh1d[i] - self.mesh1d[i - 1])), retconstfunc((-self.coeffs[i] + self.coeffs[i + 1]) / (self.mesh1d[i + 1] - self.mesh1d[i])))))
        
        seps.append((self.mesh1d[-1], (retconstfunc((-self.coeffs[-2] + self.coeffs[-1]) / (self.mesh1d[-1] - self.mesh1d[-2])), lambda x: 0)))
        return PiecewiseFunction(seps)


                
def make_hat_functions(mesh1d):
    # Assume mesh1d is sorted in ascending order
    n = len(mesh1d) - 1
    hats = [Hat(mesh1d[0], 0, mesh1d[1] - mesh1d[0])]
    for i in range(1, n):
        hatfunc = Hat(mesh1d[i], mesh1d[i] - mesh1d[i - 1], mesh1d[i + 1] - mesh1d[i])
        hats.append(hatfunc)
    hats.append(Hat(mesh1d[-1], mesh1d[-1] - mesh1d[-2], 0))
    return hats # Length n + 1 for consistency

def solve_poissons_1d_dirchlet_simple(mesh1d, func):
    '''
    Imposes boundary conditions of u(a) = u(b) = 0
    '''
    n = len(mesh1d) - 1
    hatfuncs = make_hat_functions(mesh1d)
    hatfuncs[0] = Hat(mesh1d[0], 0, 0) # Trial function satisfies v(a) = 0
    hatfuncs[-1] = Hat(mesh1d[-1], 0, 0) # Trial function satisfies v(b) = 0

    # Assemble n-1 x n-1 stiffness matrix A
    A = []
    for i in range(1, n):
        vec = [0 for _ in range(n - 1)]
        vec[i - 1] = (1 / (mesh1d[i] - mesh1d[i - 1]) + 1 / (mesh1d[i + 1] - mesh1d[i]))
        if i != n - 1:
            vec[i] = -1 / (mesh1d[i + 1] - mesh1d[i])
        if i != 1:
            vec[i - 2] = -1 / (mesh1d[i] - mesh1d[i - 1])
        A.append(vec)
    # Assemble n-1 vector b
    b = []
    for i in range(1, n):
        item = 0
        # First half - Simpson's rule
        item += ((mesh1d[i] - mesh1d[i - 1]) / 6) * (2 * func((mesh1d[i] + mesh1d[i - 1]) / 2) + func(mesh1d[i]))
        # Second half
        item += ((mesh1d[i + 1] - mesh1d[i]) / 6) * (func(mesh1d[i]) + 2 * func((mesh1d[i + 1] + mesh1d[i]) / 2))
        b.append(item)

    # Solve for the coefficients
    X = np.linalg.solve(A, b)
    X = [0] + list(X) + [0]
    return CompositeHat(hatfuncs, X, mesh1d, func)

def refine_mesh(old_solution, mesh1d, threshold=0.2, visits=1):
    #diffs = old_solution.difflist()
    running_mean = 0
    M2 = 0
    readings = 0
    new_mesh = list(dcp(mesh1d))
    n = dcp(len(mesh1d))
    print("Initial number of nodes", n)
    while readings < int(n * visits):
        index = random.randint(0, n - 2)
        # Since -u'' = f, over each element the residual f + u'' = 2f >= 0. Therefore int(f) over an element is roughly proportional to the error in that element.
        a = old_solution.func(mesh1d[index])
        b = old_solution.func(mesh1d[index + 1])
        h = mesh1d[index + 1] - mesh1d[index]
        t = (a * a + b * b) * h / 2
        residual = h * np.sqrt(t)
        readings += 1
        delta = residual - running_mean
        running_mean += delta / readings
        delta2 = residual - running_mean
        M2 += (delta * delta2)
        variance = M2 / readings
        prob = stats.norm.cdf(residual, running_mean, variance ** (1/2))
        if (prob < threshold) or (prob > 1 - threshold):
            new_mesh[index] = (mesh1d[index], 0.5 * (mesh1d[index] + mesh1d[index + 1]))
    real_new_mesh = []
    for _ in new_mesh:
        if type(_) == tuple:
            real_new_mesh += [__ for __ in _]
        else:
            real_new_mesh.append(_)
    print("Final number of nodes", len(real_new_mesh))
    return real_new_mesh
    


if __name__ == "__main__":
    # Example: f(x) = sin(5x), a = -2, b = 3, uniform mesh with step size = 0.5
    start = time.time()
    runs = 10
    step = 0.5
    a = -2
    b = 3
    mesh1d = np.arange(a, b + step, step)
    func = lambda x: np.sin(5 * x)
    A = (np.sin(10) + np.sin(15)) / 125
    B = 2 * A - np.sin(10) / 25
    actual = lambda x: np.sin(5 * x) / 25 - A * x - B
    X = np.arange(a, b + step/10, step/10) # Higher quality plot
    Y_actual = [actual(_) for _ in X]
    for q in range(runs):
        res = solve_poissons_1d_dirchlet_simple(mesh1d, func)
        print(time.time() - start)
        Y = [res(_) for _ in X]
        plt.plot(X, Y)
        start = time.time()
        plt.plot(X, Y_actual)
        plt.legend(["Approximate solution", "Actual solution"])
        print(time.time() - start)
        plt.scatter(mesh1d, np.zeros(len(mesh1d)))
        plt.show()
        if q != runs - 1:
            mesh1d = refine_mesh(res, mesh1d, threshold=0.15, visits=3)
            plt.clf()
        
