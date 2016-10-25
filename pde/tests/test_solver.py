import unittest
import sys
sys.path.insert(0,'..')
import pde as p
import scipy as sp

class TestProblem(unittest.TestCase):
    def setUp(self):
        self.testclass=p.solvers.solver(p.problems.problem())

    def test_f(self):
        #should return a 1x1 array
        s= self.testclass
        f = s.f(sp.array([0.]))
        self.assertIsInstance(f,sp.ndarray)
        self.assertEqual((1,1),f.shape)

    def test_df(self):
        #should return a Dx1 array
        s = self.testclass
        g = s.df(sp.array([0.]))
        self.assertIsInstance(g,sp.ndarray)
        self.assertEqual((s.D, 1), g.shape)

    def test_d2f(self):
        #should return a DxD symetric array
        s = self.testclass
        h = s.d2f(sp.array([0.]))
        self.assertIsInstance(h,sp.ndarray)
        self.assertEqual((s.D, s.D), h.shape)
        #check symmetry
        for i in xrange(s.D):
            for j in xrange(i+1,s.D):
                self.assertEqual(h[i,j],h[j,i])

if __name__=="__main__":
    unittest.main(verbosity=2)