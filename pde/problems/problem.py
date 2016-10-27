import scipy as sp
from scipy import linalg as spl

class solution(object):
    def __init__(self,D,u,du,d2u):
        self.D = D
        self.u=u
        self.du = du
        self.d2u = d2u
        self.soln_known=True

    def __call__(self,X,dv=0):
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
            R[0,i]=self.u(X[:,i])
        if d:
            for i in xrange(n):
                R[1:1+self.D,i]=self.du(X[:,i])
        if d2:
            for i in xrange(n):
                R[1+self.D:m,i]=self.d2u(X[:,i])
        return R

class solution_var(solution):
    def __call__(self,X,dv=0,var=False):
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
        if not var:
            R=sp.empty([m,n])
            for i in xrange(n):
                R[0,i]=self.u(X[:,i])
            if d:
                for i in xrange(n):
                    R[1:1+self.D,i]=self.du(X[:,i])
            if d2:
                for i in xrange(n):
                    R[1+self.D:m,i]=self.d2u(X[:,i])
            return R
        else:
            R = sp.empty([m, n])
            V = sp.empty([m, n])
            for i in xrange(n):
                u,v = self.u(X[:, i],var=True)
                R[0, i] = u
                V[0, i] = v
            if d:
                for i in xrange(n):
                    u,v = self.du(X[:, i],var=True)
                    R[1:1 + self.D, i] = u
                    V[1:1 + self.D, i] = v

            if d2:
                for i in xrange(n):
                    u,v = self.d2u(X[:, i],var=True)
                    R[1 + self.D:m, i] = u
                    V[1 + self.D:m, i] = v
            return R,V


class problem(object):
    """
    solution and observation generator for a 2nd order pde a*lap u + b dot div u + c*u = rhs(x)
    """
    def __init__(self,D=1):
        self.D=D
        self.a=lambda x:0.
        self.b=lambda x:sp.zeros([1,D])
        self.c=lambda x:1.
        #true value at x

        def f(X):
            return sp.zeros([1, 1])
        def df(X):
            return sp.zeros([self.D, 1])
        def d2f(X):
            return sp.zeros([self.D, self.D])
        self.soln = solution(1,f,df,d2f)
        return

    def rhs(self,X):
        """
        true value at X
        :param X:
        :return:
        """
        assert X.shape == (self.D,), "query is {} should be ({},)".format(X.shape, self.D)
        return 0.

    def lhs(self,X):
        """
        lhs values at x
        :param X:
        :return:
        """
        assert X.shape == (self.D,), "query is {} should be ({},)".format(X.shape, self.D)
        #R=self.a(X)*self.d2f(X).sum()+self.b(X).dot(self.df(X))[0,0]+self.c(X)*self.f(X)[0,0]
        return self.a(X), self.b(X), self.c(X)



class linear(problem):
    def __init__(self,dmleft=-1.,dmright=1.,bcleft=0.,bcright=0.):
        super(linear, self).__init__(1)
        self.D=1

        self.dmleft=dmleft
        self.dmright=dmright
        self.bcleft=bcleft
        self.bcright=bcright

    def plot(self,axis,n=400,col='b'):
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
    1d poisson with the rhs=0 -pc d2u/dx = rhs
    """
    def __init__(self, dmleft=-1., dmright=1., bcleft=0., bcright=0.,pc=1.):
        super(poisson1d, self).__init__(dmleft, dmright, bcleft, bcright)
        self.a=lambda x:-pc
        self.b = lambda x:sp.zeros([1, 1])
        self.c = lambda x:0.

        def f(X):
            x=X[0]
            u = self.bcleft + (self.bcright-self.bcleft)*(x-self.dmleft)/(self.dmright-self.dmleft)
            return sp.array([[u]], dtype=sp.float64)

        def df(X):
            x=X[0]
            du = (self.bcright-self.bcleft)/(self.dmright-self.dmleft)
            return sp.array([[du]], dtype=sp.float64)

        def d2f(X):
            return sp.zeros([1,1])

        self.soln = solution(1, f, df, d2f)


class poisson1dquad(poisson1d):
    """
    1d poisson witht he rhs=p2*x^2+p1*x+p0
    """
    def __init__(self, dmleft=-1., dmright=1., bcleft=0., bcright=0., pc= 1., p2=0.,p1=0.,p0=0.):
        super(poisson1dquad, self).__init__(dmleft, dmright, bcleft, bcright, pc)
        self.p2=p2
        self.p1=p1
        self.p0=p0
        #gen the solution
        self.gensoln()

    def gensoln(self):
        # diide by pc to normalise
        pc = -self.a(0.)
        a=self.p2/pc
        b=self.p1/pc
        c=self.p0/pc
        self.rhspara=[self.p2,self.p1,self.p0]
        M = sp.array([[self.dmleft,1.],[self.dmright,1.]])
        x0=self.dmleft
        x1=self.dmright
        y = sp.array([[-self.bcleft-a*(x0**4)/12.-b*(x0**3)/6.-c*(x0**2)/2.],[-self.bcright-a*(x1**4)/12.-b*(x1**3)/6.-c*(x1**2)/2.]])
        p = spl.solve(M,y)
        #print [M,y,p]
        sln = sp.array([-a/12.,-b/6.,-c/2.,-p[0,0],-p[1,0]])

        def f(X):
            x = X[0]
            u = sln[0] * x ** 4 + sln[1] * x ** 3 + sln[2] * x ** 2 + sln[3] * x + sln[4]
            return sp.array([[u]], dtype=sp.float64)

        def df(X):
            x = X[0]
            du = 4. * sln[0] * x ** 3 + 3. * sln[1] * x ** 2 + 2. * sln[2] * x + sln[3]
            return sp.array([[du]], dtype=sp.float64)

        def d2f(X):
            x = X[0]
            du = 12. * sln[0] * x ** 2 + 6. * sln[1] * x + 2. * sln[2]
            return sp.array([[du]], dtype=sp.float64)

        self.soln = solution(1, f, df, d2f)


    def rhs(self,X):
        """
        true value at X
        :param X:
        :return:
        """
        assert X.shape == (self.D,), "query is {} should be ({},)".format(X.shape, self.D)
        x=X[0]
        return self.rhspara[0]*x**2 + self.rhspara[1]*x + self.rhspara[2]


