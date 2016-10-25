import scipy as sp

class solver(object):
    """base class for solving a pde"""
    def __init__(self,problem):
        self.problem=problem
        self.D=problem.D

    def solve(self):
        pass

    def f(self,X):
        """
        true value at X
        :param X:
        :return:
        """
        return sp.zeros([1,1],dtype=sp.float64)

    def df(self,X):
        """
        derivatives at X
        :param X:
        :return:
        """
        return sp.zeros([self.D, 1])

    def d2f(self,X):
        """
        second derivatives at X, ordered by leading diagonal then second. A 3d input would give
        [dx2, dy2, dz2, dxdy, dydz,  dxdz]
        :param X:
        :return:
        """
        return sp.zeros([self.D, self.D])