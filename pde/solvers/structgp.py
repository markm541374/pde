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


class poisson_1d_solver(solver.linearsolver_var):
    """
    this exploits the regularity of the space
    """
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

        #K matrix is structured as
        #
        # A | B
        # B'| D
        # where A is symmetric toeplitz with leadinf row Ac
        #
        # Kinv is then
        #
        # A^ + A^ B (D-B'A^ B)^ B' A^  |  -A^ B (D-B'A^ B)^
        # -(D-B'A^ B)^ B' A^           |   (D-B'A^ B)^
        #
        # Let A^B = R
        #     B'A^ = R'
        #
        # A^ + R (D-B' R)^ R'  |  -R (D-B' R)^
        # -(D-B'A^ B)^ R'      |   (D-B' R)^
        #
        # Let D-B'R = Q
        #
        # A^ + R Q^ R'  |  -R Q^
        # -Q^ R'        |   Q^
        #
        #
        # final rename
        #
        # A^ - RV' | V
        # V'       | Q^
        #
        # Also note
        #
        #
        #det(A | B
        #    B'| D) = det(A)*det(D-B'A^B)
        #           = det(A)*det(Q)
        #

        k = sqk(1.0011159368491009, 0.7256396362752614)

        def buildmats(k):
            Ac = sp.empty([1, n])
            B = sp.empty([n, 2])
            D = sp.empty([2, 2])
            for i in xrange(n):
                Ac[0,i]=k.d4(X[0, 0], X[0, i])
            Ac[0,0]+=1e-9
            for i in xrange(n):
                B[i, 0] = k.d2(self.dmleft, X[0,i])
                B[i, 1] = k.d2(self.dmright, X[0,i])

            D[0,0]=k(self.dmleft, self.dmleft)+1e-9
            D[1,1]=k(self.dmright, self.dmright)+1e-9
            D[1,0]=D[0,1]=k(self.dmright, self.dmleft)

            R = sp.linalg.solve_toeplitz(Ac,B)
            Q = D-B.T.dot(R)

            V = -sp.linalg.solve(Q,R.T,sym_pos=True).T
            return Ac,R,Q,V

        def Ksolve(Ac,R,Q,V,Z):
            W = sp.empty(Z.shape)
            W[:n, :] = sp.linalg.solve_toeplitz(Ac, Z[:n, :]) + -R.dot(V.T.dot(Z[:n, :])) + V.dot(Z[n:, :])
            W[n:, :] = V.T.dot(Z[:n, :]) + sp.linalg.solve(Q, Z[n:, :])
            return W

        def llk(x):
            A = 10 ** x[0]
            l = 10 ** x[1]
            k = sqk(A, l)
            Ac, R, Q, V = buildmats(k)
            lk = -0.5 * Y.T.dot(Ksolve(Ac, R, Q, V ,Y)) - 0.5 * (slogdet(spl.toeplitz(Ac))[1]+slogdet(Q)[1])

            pr = -0.5 * x[0] ** 2 - 0.5 * x[1] ** 2
            return -lk - pr
        opt = spo.minimize(llk, [0., sp.log10(0.4)], method='Nelder-Mead', options={'xatol': 0.1})
        # print opt
        A, l = [10 ** i for i in opt.x]
        logger.info('map hyperparameters {} under lognormal(0,1) prior'.format((A, l)))
        k = sqk(A, l)

        Ac, R, Q, V = buildmats(k)
        KiY = Ksolve(Ac,R,Q,V,Y)
        self.solvetime = timer() - t0
        logger.info('solvetime = {}'.format(self.solvetime))
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
                v = k(x,x)-Kxy.T.dot(Ksolve(Ac,R,Q,V,Kxy))
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
                v = -k.d2(x,x)-Kxy.T.dot(Ksolve(Ac,R,Q,V,Kxy))
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
                v = k.d4(x,x)-Kxy.T.dot(Ksolve(Ac,R,Q,V,Kxy))
                #print k.d4(x,x),Kxy.T.dot(spl.cho_solve(C,Kxy))
                #print Kxy
                return sp.array([[u]]),sp.array([[v]])

        self.soln = solution_var(self.D, u, du, d2u)
        return
        return

