import scipy as sp
from pde.problems import solution as solution
import logging
logger = logging.getLogger(__name__)

class solver(object):
    """base class for solving a pde"""
    def __init__(self,problem):
        self.problem=problem
        self.D=problem.D
        def f(X):
            return sp.zeros([1, 1])

        def df(X):
            return sp.zeros([self.D, 1])

        def d2f(X):
            return sp.zeros([self.D, self.D])

        self.soln = solution(self.D,f,df,d2f)
        return

    def solve(self):
        pass

class linearsolver(solver):
    def __init__(self,problem):
        assert problem.D==1
        super(linearsolver, self).__init__(problem)
        self.dmleft = problem.dmleft
        self.dmright = problem.dmright
        return

    def plot(self,axis,n=400,col='r'):
        assert(len(axis)>=3)
        xaxis = sp.linspace(self.dmleft,self.dmright,n).reshape([1,n])
        F = self.soln(xaxis,dv=2)
        axis[0].plot(xaxis[0,:],F[0,:],col)
        axis[1].plot(xaxis[0,:],F[1,:],col)
        axis[2].plot(xaxis[0,:],F[2,:],col)
        return

    def ploterr(self,axis,n=400,col='r'):
        assert(len(axis)>=3)
        xaxis = sp.linspace(self.dmleft,self.dmright,n).reshape([1,n])
        F = self.soln(xaxis,dv=2)-self.problem.soln(xaxis,dv=2)
        axis[0].plot(xaxis[0,:],F[0,:],col)
        axis[1].plot(xaxis[0,:],F[1,:],col)
        axis[2].plot(xaxis[0,:],F[2,:],col)
        return

    def errstats(self,n=10000):
        xaxis = sp.random.uniform(self.dmleft, self.dmright, n).reshape([1, n])
        F = (self.soln(xaxis, dv=2) - self.problem.soln(xaxis, dv=2))**2
        means = F.sum(axis=1)/n
        maxs = F.max(axis=1)
        logger.info('errors are mean^2 {} max^2 {}'.format(means,maxs))
        return means,maxs