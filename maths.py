import random


class Gen1DPoly:

    # Generic 1D polynominal.
    # Form: a_0 + a_1 * x + a_2 * x^2 + ...
    # Inputs:
    # coeffs - coefficients, None by default, array of floats if overriden
    # empty - None by default, zero polynominal of degree empty if overriden

    def __init__(self, coeffs = None, empty = None):
        if coeffs != None:
            self.d = len(coeffs) - 1
            self.coeffs = coeffs
        elif empty:
            self.coeffs = [0] * (empty + 1)
            self.d = empty

    def randomise(self):
        self.coeffs = [random.random() for coeff in self.coeffs]

    def __add__(self, other):
        # Check degrees
        maxd = max(self.d, other.d)
        self = self.coerce(maxd)
        other = other.coerce(maxd)
        return Gen1DPoly([self.coeffs[i] + other.coeffs[i] for i in range(self.d + 1)])
    
    def __sub__(self, other):
        # Check degrees
        maxd = max(self.d, other.d)
        self = self.coerce(maxd)
        other = other.coerce(maxd)
        return Gen1DPoly([self.coeffs[i] - other.coeffs[i] for i in range(self.d + 1)])
    
    def __mul__(self, other):
        if isinstance(other, Gen1DPoly):
            newcoeffs = [0] * (self.d + other.d + 1)
            for i in range(len(self.coeffs)):
                for j in range(len(other.coeffs)):
                    newcoeffs[i + j] += self.coeffs[i] * other.coeffs[j]
        else:
            newcoeffs = [other * coef for coef in self.coeffs]
        return Gen1DPoly(newcoeffs)
    
    def value(self, x):
        return sum([self.coeffs[i] * (x ** i) for i in range(self.d+1)])
    
    def derivative(self):
        return Gen1DPoly([self.coeffs[i] * i for i in range(1, self.d+1)]).coerce(self.d)

    def integral(self):
        # Returns the antiderivative
        return Gen1DPoly([0] + [self.coeffs[i] / (i+1) for i in range(self.d + 1)])
    
    def coerce(self, d):
        # Coerce to a polynominal of degree d, with missing coefficents = 0
        newcoeffs = self.coeffs + [0] * (d - self.d)
        return Gen1DPoly(newcoeffs)
    
    def similar(self, other, lower, upper):
        # Calculates the integral of (f(x) -g(x))^2 from lower < x < upper
        theint = (self - other) * (self - other)
        res = theint.integral()
        return res.value(upper) - res.value(lower)

    
class Gen2DPoly:

    # Generic 2D polynominal.
    # Form: a_00 + a_10 * x + a_01 * y + a_20 * x^2 + a_11 * x * y + a_02 * y^2 + ...
    # Inputs:
    # coeffs - coefficients, None by default, array of arrays if overriden
    # empty - None by defaut, generates zero polynominal of degree empty if overriden

    def __init__(self, coeffs = None, empty = None):
        if coeffs != None:
            self.d = len(coeffs) - 1
            self.coeffs = coeffs
        elif empty:
            self.coeffs = [[0] * i for i in range(1, empty + 2)]
            self.d = empty

    def randomise(self):
        self.coeffs = [[random.random() for coef in coeff] for coeff in self.coeffs]

    def __add__(self, other):
        # Check degrees
        maxd = max(self.d, other.d)
        self = self.coerce(maxd)
        other = other.coerce(maxd)
        return Gen2DPoly([[self.coeffs[i][j] + other.coeffs[i][j] for j in range(len(self.coeffs[i]))] for i in range(len(self.coeffs))])
    
    def __sub__(self, other):
        # Check degrees
        maxd = max(self.d, other.d)
        self = self.coerce(maxd)
        other = other.coerce(maxd)
        return Gen2DPoly([[self.coeffs[i][j] - other.coeffs[i][j] for j in range(len(self.coeffs[i]))] for i in range(len(self.coeffs))])
    
    def __mul__(self, other):
        if isinstance(other, Gen2DPoly):
            newcoeffs = [[0] * i for i in range(1, 2 * (self.d + 1))]
            for i in range(len(self.coeffs)):
                for j in range(len(self.coeffs[i])):
                    for k in range(len(other.coeffs)):
                        for l in range(len(other.coeffs[k])):
                            newcoeffs[i + k][j + l] += self.coeffs[i][j] * other.coeffs[k][l]
        else:
            newcoeffs = [[other * coef for coef in coeff] for coeff in self.coeffs]
        return Gen2DPoly(newcoeffs)
    __rmul__ = __mul__

    def coerce(self, d):
        # Coerce to a polynominal of degree d, with missing coefficiets = 0
        if self.d >= d:
            # Truncate
            return Gen2DPoly(self.coeffs[:d + 1])
        newcoeffs = [[0] * i for i in range(1, d+2)]
        for i in range(len(self.coeffs)):
            for j in range(len(self.coeffs[i])):
                newcoeffs[i][j] = self.coeffs[i][j]
        return Gen2DPoly(newcoeffs)

    def create1DPoly(self, x = None, y = None):
        newcoeffs = [0] * (self.d + 1)
        if x == None:
            for pow in range(len(self.coeffs)):
                for step in range(len(self.coeffs[pow])):
                    newcoeffs[pow - step] += self.coeffs[pow][step] * y ** (step)

        elif y == None:
            for pow in range(len(self.coeffs)):
                for step in range(len(self.coeffs[pow])):
                    newcoeffs[step] += self.coeffs[pow][step] * x ** (pow - step)

        return Gen1DPoly(newcoeffs)

    def value(self, x, y):
        return self.create1DPoly(y=y).value(x)
    
    def partialx(self):
        newcoeffs = [[0] * (i + 1) for i in range(self.d)]
        for pow in range(len(self.coeffs)):
            for step in range(len(self.coeffs[pow])):
                if step == pow:
                    continue
                else:
                    newcoeffs[pow - 1][step] += self.coeffs[pow][step] * (pow - step)
        return Gen2DPoly(newcoeffs).coerce(self.d)
    
    def partialy(self):
        newcoeffs = [[0] * (i) for i in range(1,self.d + 1)]
        for pow in range(len(self.coeffs)):
            for step in range(len(self.coeffs[pow])):
                if step == 0:
                    continue
                else:
                    newcoeffs[pow - 1][step - 1] += self.coeffs[pow][step] * (step)
        return Gen2DPoly(newcoeffs).coerce(self.d)

    def grad(self):
        return VectorFieldPoly2D(self.partialx(), self.partialy())
    
class VectorFieldPoly2D:

    # Polynominal 2D Vector Field
    # Inputs:
    # fi - 2D Polynominal in x, y describing the i-component
    # fj - 2D Polynominal in x, y describing the j-component

    def __init__(self, fi, fj):
        self.fi = fi
        self.fj = fj

    def __add__(self, other):
        return VectorFieldPoly2D(self.fi + other.fi, self.fj + other.fj)
    
    def __sub__(self, other):
        return VectorFieldPoly2D(self.fi - other.fi, self.fj - other.fj)
    
    def __mul__(self, other):
        return VectorFieldPoly2D(other * self.fi, other * self.fj)
    __rmul__ = __mul__

    def abs2(self):
        return (self.fi * self.fi + self.fj * self.fj)
    
    def dot(self, other):
        if isinstance(other, VectorFieldPoly2D):
            return self.fi * other.fi + self.fj * other.fj
        else:
            # Probably a 2x2 matrix            
            return VectorFieldPoly2D(other[0][0] * self.fi + other[1][0] * self.fj, other[0][1] * self.fi + other[1][1] * self.fj)
    
    def grad(self):
        # Using the tensor definition. Returns a matrix (list of lists)
        return [[self.fi.partialx(), self.fj.partialx()], [self.fi.partialy(), self.fj.partialy()]]
    
    def jacobian(self):
        return [[self.fi.partialx(), self.fi.partialy()], [self.fj.partialx(), self.fj.partialy()]]

    def divergence(self):
        return self.fi.partialx() + self.fj.partialy()
    
    def laplacian(self):
        return VectorFieldPoly2D(self.fi.partialx().partialx() + self.fi.partialy().partialy(), self.fj.partialx().partialx() + self.fj.partialy().partialy())
    
    


#quad = Gen1DPoly([6,4,2])
#quad2d = Gen2DPoly([[6],[3,6],[8,1,8]])
#quad22d = Gen2DPoly([[2],[4,5],[-2,-9,-12]])

#vec2d = VectorFieldPoly2D(quad2d, quad22d)
