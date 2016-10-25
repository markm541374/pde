import scipy as sp
from scipy import linalg as spl

def pde(paras):
    return paras[0](**paras[1])

class problem(object):
    """
    solution and observation generator for a 2nd order pde a*lap u + b dot div u + c*u = rhs(x)
    """
    def __init__(self,D=1):
        self.D=D
        self.a=0.
        self.b=sp.zeros([1,D])
        self.c=1.
        #true value at x
        return

    def rhs(self,X):
        """
        true value at X
        :param X:
        :return:
        """
        assert X.shape == (self.D,), "query is {} should be ({},)".format(X.shape, self.D)
        return 0.

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

    def sf(self,n):
        """
        strong form observations at n locations
        :param n:
        :return:
        """
        return [sp.zeros([self.D,n]),sp.zeros([1,n]),sp.zeros([0,0]),sp.zeros([0,0])]

    def wf_int(self,n):
        """
        weak form observations at n locations interpolated from strong form
        :param n:
        :return:
        """
        return

    def wf_ex(self,n):
        """
        weak form observations at n locations exactly calculated
        :param n:
        :return:
        """
        return

    def soln(self,X,dv=0):
        """
        return the true solution with derivatives at X query set
        :param X:
        :param d: derivatives
        :param d2: second derivatives
        :return:
        """
        if dv==0:
            d,d2=False,False
        elif dv==1:
            d,d2 =True,False

        elif dv==2:
            d,d2=True,True
        else:
            raise NotImplementedError("dv={}! derivative higher than 2nd not implemeted".format(dv))


        m=1+d*self.D+d2*(self.D*(self.D+1)/2)
        [D_,n]=X.shape
        assert D_==self.D, "query is {}x{} should be {}x*".format(D_,n,self.D)
        R=sp.empty([m,n])
        for i in xrange(n):
            R[0,i]=self.f(X[:,i])
        if d:
            for i in xrange(n):
                R[1:1+self.D,i]=self.df(X[:,i])
        if d2:
            for i in xrange(n):
                R[1+self.D:m,i]=self.d2f(X[:,i])
        return R

    def lhs(self,X):
        """
        lhs value at x
        :param X:
        :return:
        """
        assert X.shape == (self.D,), "query is {} should be ({},)".format(X.shape, self.D)
        R=self.a*self.d2f(X).sum()+self.b.dot(self.df(X))[0,0]+self.c*self.f(X)[0,0]
        return R



class linear(problem):
    def __init__(self,dmleft=-1.,dmright=1.,bcleft=0.,bcright=0.):
        super(linear, self).__init__(1)
        self.D=1

        self.dmleft=dmleft
        self.dmright=dmright
        self.bcleft=bcleft
        self.bcright=bcright

    def plot(self,axis,n=100,col='b'):
        assert(len(axis)>=3)
        xaxis = sp.linspace(self.dmleft,self.dmright,n).reshape([1,n])
        F = self.soln(xaxis,dv=2)
        axis[0].plot(xaxis[0,:],F[0,:],col)
        axis[1].plot(xaxis[0,:],F[1,:],col)
        axis[2].plot(xaxis[0,:],F[2,:],col)
        return

    def sf(self,n):
        """
        strong form observations at n locations. two boundary conditions and n-2 rhs observations
        :param n:
        :return:
        """
        X = sp.linspace(self.dmleft,self.dmright,n-2).reshape([1,n-2])
        R = sp.empty([1,n-2])
        for i in xrange(n-2):
            R[0,i]=self.rhs(X[:,i])
        B = sp.array([[self.dmleft,self.dmright]])
        Z = sp.array([[self.bcleft,self.bcright]])
        return X,R,B,Z

class poisson1d(linear):
    """
    1d poisson witht he rhs=0
    """
    def __init__(self, dmleft=-1., dmright=1., bcleft=0., bcright=0.):
        super(poisson1d, self).__init__(dmleft, dmright, bcleft, bcright)
        self.a=-1.
        self.b = sp.zeros([1, 1])
        self.c = 0.

    def f(self,X):
        assert X.shape == (self.D,), "query is {} should be ({},)".format(X.shape, self.D)
        x=X[0]
        u = self.bcleft + (self.bcright-self.bcleft)*(x-self.dmleft)/(self.dmright-self.dmleft)
        return sp.array([[u]], dtype=sp.float64)

    def df(self,X):
        assert X.shape == (self.D,), "query is {} should be ({},)".format(X.shape, self.D)
        x=X[0]
        du = (self.bcright-self.bcleft)/(self.dmright-self.dmleft)
        return sp.array([[du]], dtype=sp.float64)

class poisson1dquad(poisson1d):
    """
    1d poisson witht he rhs=p2*x^2+p1*x+p0
    """
    def __init__(self, dmleft=-1., dmright=1., bcleft=0., bcright=0.,p2=0.,p1=0.,p0=0.):
        super(poisson1dquad, self).__init__(dmleft, dmright, bcleft, bcright)
        a=p2
        b=p1
        c=p0
        self.rhspara=[a,b,c]
        M = sp.array([[dmleft,1.],[dmright,1.]])
        x0=dmleft
        x1=dmright
        y = sp.array([[-bcleft-a*(x0**4)/12.-b*(x0**3)/6.-c*(x0**2)/2.],[-bcright-a*(x1**4)/12.-b*(x1**3)/6.-c*(x1**2)/2.]])
        p = spl.solve(M,y)
        #print [M,y,p]
        self.sln = sp.array([-a/12.,-b/6.,-c/2.,-p[0,0],-p[1,0]])
        #print self.sln

    def rhs(self,X):
        """
        true value at X
        :param X:
        :return:
        """
        assert X.shape == (self.D,), "query is {} should be ({},)".format(X.shape, self.D)
        x=X[0]
        return self.rhspara[0]*x**2 + self.rhspara[1]*x + self.rhspara[2]


    def f(self,X):
        assert X.shape == (self.D,), "query is {} should be ({},)".format(X.shape, self.D)
        x=X[0]
        u = self.sln[0]*x**4 + self.sln[1]*x**3 + self.sln[2]*x**2 + self.sln[3]*x + self.sln[4]
        return sp.array([[u]], dtype=sp.float64)

    def df(self,X):
        assert X.shape == (self.D,), "query is {} should be ({},)".format(X.shape, self.D)
        x=X[0]
        du = 4.*self.sln[0]*x**3 + 3.*self.sln[1]*x**2 + 2.*self.sln[2]*x + self.sln[3]
        return sp.array([[du]], dtype=sp.float64)

    def d2f(self,X):
        assert X.shape == (self.D,), "query is {} should be ({},)".format(X.shape, self.D)
        x=X[0]
        du = 12.*self.sln[0]*x**2 + 6.*self.sln[1]*x + 2.*self.sln[2]
        return sp.array([[du]], dtype=sp.float64)