import pde
from matplotlib import pyplot as plt

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

f,ap = plt.subplots(3)
f,ae = plt.subplots(3)

prob = pde.problems.instpoisson1d.quad
prob.plot(ap)

solver = pde.solvers.classic.poisson_pwlinear_1d_solver(prob)
solver.solve(n=3)
solver.plot(ap,col='r')
solver.ploterr(ae,col='r',logabs=True)
solver.errstats()

#solver = pde.solvers.fullgp.poisson_1d_solver(prob)
#solver.solve(n=3)
#solver.plot(ap,col='g')
#solver.ploterr(ae,col='g',logabs=True)
#solver.errstats()

solver = pde.solvers.structgp.poisson_1d_solver(prob)
solver.solve(n=3)
solver.plot(ap,col='c')
solver.ploterr(ae,col='c',logabs=True)
solver.errstats()

plt.show()