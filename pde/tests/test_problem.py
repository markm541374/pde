import unittest
import sys
sys.path.insert(0,'..')
import scipy as sp

import pde.problems as p


class TestProblem(unittest.TestCase):
    def setUp(self):
        self.testclass=p.problem()

    def test_rhs(self):
        #should return a 1x1 array
        s= self.testclass
        f = s.rhs(sp.array([0.]))
        self.assertIsInstance(f,float)

    def test_lhs(self):
        #should return a 1x1 array
        s= self.testclass
        a,b,c = s.lhs(sp.array([0.]))
        self.assertIsInstance(a,float)
        self.assertIsInstance(b, sp.ndarray)
        self.assertEqual((s.D, 1), b.shape)
        self.assertIsInstance(c, float)

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

    def test_eqsides(self):
        pass


class Testlinear(TestProblem):
    def setUp(self):
        super(Testlinear, self).setUp()
        self.testclass=p.linear()

    #def test_sf(self):
    #    s = self.testclass
    #    n=40
    #    X, R, B, Z = s.sf(n)
     #   #n-2 rhs observations
    #    self.assertIsInstance(X, sp.ndarray)
    #    self.assertEqual((1, n - 2), X.shape)
    #    self.assertIsInstance(R, sp.ndarray)
     #   self.assertEqual((1, n - 2), R.shape)
    #    #2 bc
     #   self.assertIsInstance(B, sp.ndarray)
    #    self.assertEqual((1, 2), B.shape)
     #   self.assertIsInstance(Z, sp.ndarray)
     #   self.assertEqual((1, 2), Z.shape)

    def test_eqsides(self):
        """
        check the rhs and lh are equal at sup points
        :return:
        """
        s = self.testclass
        m = 40
        X = sp.linspace(s.dmleft,s.dmright,m).reshape([1,m])

        RHS=sp.empty([1,m])
        LHS=sp.empty([1,m])
        for i in xrange(m):
            RHS[0,i]=s.rhs(X[:,i])
            a,b,c = s.lhs(X[:, i])
            LHS[0,i]=s.soln.d2u(X[:, i]).sum()*a + b.dot(s.soln.du(X[:, i]))[0,0]+c*s.soln.u(X[:,i])[0,0]
        self.assertTrue(sp.allclose(RHS,LHS))

class Testpoisson1d(Testlinear):
    def setUp(self):
        super(Testpoisson1d, self).setUp()
        self.testclass=p.poisson1d()

class Testpoisson1dquad(Testlinear):
    def setUp(self):
        super(Testpoisson1dquad, self).setUp()
        self.testclass=p.poisson1dquad()

if __name__=="__main__":
    unittest.main(verbosity=2)