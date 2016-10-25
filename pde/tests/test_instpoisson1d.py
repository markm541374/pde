import unittest
import test_problem as tp
import sys
sys.path.insert(0,'..')



import pde.problems

class test_unforced(tp.Testpoisson1d):
    def setUp(self):
        super(test_unforced, self).setUp()
        self.testclass=pde.problems.instpoisson1d.unforced

class test_quad(tp.Testpoisson1d):
    def setUp(self):
        super(test_quad, self).setUp()
        self.testclass=pde.problems.instpoisson1d.quad

if __name__=="__main__":
    unittest.main(verbosity=2)