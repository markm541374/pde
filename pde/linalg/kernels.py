import scipy as sp

class kernel(object):
    def __init__(self):
        pass

    def __call__(self, x1,x2):
        return 0.

class sqexp1d(kernel):
    """
    1d squared exponential kernel
    k = A exp(-0.5(x2-x1)**2/l**2)
    dn methods are k( x1, d^n x2/dx^n) the caller is responsible for sign changes for other derivatives
    """
    def __init__(self,A,l):
        super(sqexp1d,self).__init__()
        self.hyperpara = (A,l,)
        self.A=A
        self.overl2 = 1./l**2
        return

    def __call__(self, x1, x2):
        return self.A*sp.exp(-0.5*self.overl2*(x2-x1)**2)

    def d1(self,x1,x2):
        return -(x2-x1)*self.overl2*self.A*sp.exp(-0.5*self.overl2*(x2-x1)**2)

    def d2(self,x1,x2):
        return self.overl2*(self.overl2*(x2-x1)**2 - 1)*self.A*sp.exp(-0.5*self.overl2*(x2-x1)**2)

    def d3(self,x1,x2):
        x = (x2 - x1)
        return (self.overl2**2)*x*(3-(x**2)*self.overl2)*self.A*sp.exp(-0.5*self.overl2*(x2-x1)**2)

    def d4(self,x1, x2):
        x=(x2-x1)
        return (self.overl2**2)*(3-6*self.overl2*(x**2)+(x**4)*self.overl2**2)*self.A*sp.exp(-0.5*self.overl2*(x2-x1)**2)