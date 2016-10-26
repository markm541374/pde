import scipy as sp
from scipy import linalg as spl
from pde.problems import solution as solution
import solver
import sys
from timeit import default_timer as timer
import logging
logger = logging.getLogger(__name__)

class poisson_pwlinear_1d_solver(solver.linearsolver):
    def __init__(self,problem):
        super(poisson_pwlinear_1d_solver, self).__init__(problem)

    def solve(self,n=20,kappa=1e6):
        t0 = timer()
        #n support points including boundary points
        X = sp.linspace(self.dmleft,self.dmright,n).reshape([1,n])
        self.xi_x=X
        #stiffness at mid of points
        #print a
        #interval size
        h = (self.dmright-self.dmleft)/(n-1)

        A = sp.zeros([2,n])
        for i in xrange(n-1):
            a=self.problem.lhs(0.5*(X[:,i]+X[:,i+1]))[0]
            A[1,i]+=-a/h
            A[1,i+1]+=-a/h
            A[0,i+1]-=-a/h
        A[1,0]+=kappa
        A[1,n-1]+=kappa

        b = sp.zeros([n,1])
        b[0,0]=self.problem.rhs(X[:,0])*h*0.5 + kappa*self.problem.bcleft
        b[n-1, 0]= self.problem.rhs(X[:, n-1]) * h * 0.5 + kappa * self.problem.bcright
        for i in xrange(1,n-1):
            b[i,0]=self.problem.rhs(X[:,i])*h
        self.xi = spl.solveh_banded(A, b,lower=False)

        self.solvetime = timer()-t0
        logger.info('solvetime = {}'.format(self.solvetime))
        def xtoif(x):
            i,r = divmod(min(x,self.dmright-1e-9)-self.dmleft,h)
            return int(i),r/h
        def u(x):
            i,f = xtoif(x[0])
            u=(1.-f)*self.xi[i,0] +f*self.xi[i+1,0]
            return sp.array([[u]])
        def du(x):
            i,f = xtoif(x[0])
            du=(self.xi[i+1,0]- self.xi[i,0])/h
            return sp.array([[du]])
        def d2u(x):
            return sp.array([[0.]])

        self.soln = solution(self.D, u, du, d2u)
        return