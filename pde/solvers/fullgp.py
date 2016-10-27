import scipy as sp
from scipy import linalg as spl
from scipy import optimize as spo
from numpy.linalg import slogdet as slogdet
from pde.problems import solution_var as solution_var
from pde.linalg.kernels import sqexp1d as sqk
import solver
import sys
from timeit import default_timer as timer
import logging
logger = logging.getLogger(__name__)


class poisson_1d_solver(solver.linearsolver):
    def __init__(self,problem):
        super(poisson_1d_solver, self).__init__(problem)

    def solve(self,n=40):
        t0 = timer()
        # n support points including boundary points
        X = sp.linspace(self.dmleft, self.dmright, n).reshape([1, n])
        # n+2 observations
        Y = sp.empty([n+2,1])
        #second derivative
        for i in xrange(n):
            Y[i,0] = self.problem.rhs(X[:,i])/self.problem.lhs(X[:,i])[0]
        Y[n,0] = self.problem.bcleft
        Y[n+1] = self.problem.bcright

        #K matrix
        K = sp.empty([n+2,n+2])
        #start values for hyperparameters
        def llk(x):
            A=10**x[0]
            l=10**x[1]
            k = sqk(A, l)
            for i in range(n):
                for j in range(i, n):
                    K[i, j] = K[j, i] = k.d4(X[0, i], X[0, j])
                K[i, i] += 1e-9
                K[n, i] = K[i, n] = k.d2(self.dmleft, X[0, i])
                K[n + 1, i] = K[i, n + 1] = k.d2(self.dmright, X[0, i])
            K[n, n] = k(self.dmleft, self.dmleft)+1e-9
            K[n + 1, n + 1] = k(self.dmright, self.dmright)+1e-9
            K[n + 1, n] = K[n, n + 1] = k(self.dmleft, self.dmright)
            lk = -0.5*Y.T.dot(spl.cho_solve(spl.cho_factor(K),Y))-0.5*slogdet(K)[1]-0.5*(n+2)*sp.log(2*sp.pi)
            pr = -0.5*x[0]**2 - 0.5*x[1]**2
            return -lk-pr
        opt = spo.minimize(llk,[0.,sp.log10(0.4)],method='Nelder-Mead',options={'xatol':0.1})
        #print opt
        A,l = [10**i for i in opt.x]
        logger.info('map hyperparameters {} under lognormal(0,1) prior'.format((A,l)))
        k=sqk(A,l)
        mapllk = llk(opt.x)

        C = spl.cho_factor(K)
        KiY = spl.cho_solve(C,Y)
        print KiY
        def u(x,var=False):
            if not var:
                u=0.
                for i in xrange(n):
                    u+=k.d2(x,X[0,i])*KiY[i,0]
                u+=k(x,self.dmleft)*KiY[n,0]
                u += k(x, self.dmright) * KiY[n+1, 0]
                return sp.array([[u]])
            else:
                Kxy = sp.empty([n+2,1])
                for i in xrange(n):
                    Kxy[i,0]=k.d2(x,X[0,i])
                Kxy[n,0]=k(x,self.dmleft)
                Kxy[n+1,0] =k(x, self.dmright)
                u = Kxy.T.dot(KiY)
                v = k(x,x)-Kxy.T.dot(spl.cho_solve(C,Kxy))
                return sp.array([[u]]),sp.array([[v]])


        def du(x,var=False):
            if not var:
                u = 0.
                for i in xrange(n):
                    u += -k.d3(x, X[0, i]) * KiY[i, 0]
                u += -k.d1(x, self.dmleft) * KiY[n, 0]
                u += -k.d1(x, self.dmright) * KiY[n + 1, 0]
                return sp.array([[u]])
            else:
                Kxy = sp.empty([n+2,1])
                for i in xrange(n):
                    Kxy[i,0]=-k.d3(x,X[0,i])
                Kxy[n,0]=-k.d1(x,self.dmleft)
                Kxy[n+1,0] =-k.d1(x, self.dmright)
                u = Kxy.T.dot(KiY)
                v = -k.d2(x,x)-Kxy.T.dot(spl.cho_solve(C,Kxy))
                return sp.array([[u]]),sp.array([[v]])

        def d2u(x,var=False):
            if not var:
                u = 0.
                for i in xrange(n):
                    u += k.d4(x, X[0, i]) * KiY[i, 0]
                u += k.d2(x, self.dmleft) * KiY[n, 0]
                u += k.d2(x, self.dmright) * KiY[n + 1, 0]
                return sp.array([[u]])
            else:
                Kxy = sp.empty([n+2,1])
                for i in xrange(n):
                    Kxy[i,0]=k.d4(x,X[0,i])
                Kxy[n,0]=k.d2(x,self.dmleft)
                Kxy[n+1,0] =k.d2(x, self.dmright)
                u = Kxy.T.dot(KiY)
                v = k.d4(x,x)-Kxy.T.dot(spl.cho_solve(C,Kxy))
                #print k.d4(x,x),Kxy.T.dot(spl.cho_solve(C,Kxy))
                #print Kxy
                return sp.array([[u]]),sp.array([[v]])

        self.soln = solution_var(self.D, u, du, d2u)
        return

    def plot(self,axis,n=400,col='r'):
        assert(len(axis)>=3)
        xaxis = sp.linspace(self.dmleft,self.dmright,n).reshape([1,n])
        F,V = self.soln(xaxis,dv=2,var=True)
        #print V
        axis[0].plot(xaxis[0,:],F[0,:],col)
        axis[0].fill_between(xaxis[0, :], F[0, :] - 2. * sp.sqrt(V[0, :]), F[0, :] + 2. * sp.sqrt(V[0,:]),
                             facecolor=col,edgecolor=col,alpha=0.1)
        axis[1].plot(xaxis[0,:],F[1,:],col)
        axis[1].fill_between(xaxis[0, :], F[1, :] - 2. * sp.sqrt(V[1, :]), F[1, :] + 2. * sp.sqrt(V[1, :]),
                             facecolor=col, edgecolor=col, alpha=0.1)
        axis[2].plot(xaxis[0,:],F[2,:],col)
        axis[2].fill_between(xaxis[0, :], F[2, :] - 2. * sp.sqrt(V[2, :]), F[2, :] + 2. * sp.sqrt(V[2, :]),
                             facecolor=col, edgecolor=col, alpha=0.1)
        return

