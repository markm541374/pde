import unittest
import sys
sys.path.insert(0,'..')
import pde as p
import scipy as sp


class TestProblem(unittest.TestCase):
    def setUp(self):
        self.testclass=p.solvers.solver(p.problems.problem())
        self.testclass.solve()

    def test_u(self):
        #should return a 1x1 array
        s= self.testclass
        f = s.soln.u(sp.array([0.]))
        self.assertIsInstance(f,sp.ndarray)
        self.assertEqual((1,1),f.shape)

    def test_du(self):
        #should return a Dx1 array
        s = self.testclass
        g = s.soln.du(sp.array([0.]))
        self.assertIsInstance(g,sp.ndarray)
        self.assertEqual((s.D, 1), g.shape)

    def test_d2u(self):
        #should return a DxD symetric array
        s = self.testclass
        h = s.soln.d2u(sp.array([0.]))
        self.assertIsInstance(h,sp.ndarray)
        self.assertEqual((s.D, s.D), h.shape)
        #check symmetry
        for i in xrange(s.D):
            for j in xrange(i+1,s.D):
                self.assertEqual(h[i,j],h[j,i])


class TestLinearSolver(TestProblem):
    def setUp(self):
        super(TestLinearSolver, self).setUp()
        self.testclass=p.solvers.linearsolver(p.problems.linear())
        self.testclass.solve()

    def test_errstats(self):
        s=self.testclass
        mn,mx = s.errstats()
        self.assertIsInstance(mn, sp.ndarray)
        self.assertEqual((3,), mn.shape)
        self.assertIsInstance(mx, sp.ndarray)
        self.assertEqual((3,), mx.shape)
        for i in xrange(3):
            self.assertGreaterEqual(mx[i],mn[i])

if __name__=="__main__":
    unittest.main(verbosity=2)