import unittest
import sys
sys.path.insert(0,'..')
import pde as p
import test_solver as ts
import scipy as sp


class test_poisson_pwlinear_1d_solver(ts.TestLinearSolver):
    def setUp(self):
        super(test_poisson_pwlinear_1d_solver, self).setUp()
        self.testclass.solve()

if __name__=="__main__":
    unittest.main(verbosity=2)