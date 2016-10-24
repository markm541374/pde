import scipy as sp

class problem:
    def __init__(self):
        self.D=1
        #true value at x
        self.f=lambda x:0
        #
        self.df=lambda x:sp.array([0]*self.D)
        self.d2f = lambda x: sp.array([0] * (self.D*(self.D+1)/2))
        return

    def f(self,X):
        """
        true value at X
        :param X:
        :return:
        """
        return 0

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
        return

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
                R[2+self.D:m,i]=self.d2f(X[:,i])
        return R

