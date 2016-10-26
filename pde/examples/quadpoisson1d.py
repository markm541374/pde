import pde
from matplotlib import pyplot as plt

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

f,a = plt.subplots(3)
prob = pde.problems.instpoisson1d.quad
prob.plot(a)

solver = pde.solvers.classic.poisson_pwlinear_1d_solver(prob)
solver.solve(n=100)
solver.plot(a)

f,a = plt.subplots(3)
solver.ploterr(a)

solver.errstats()
plt.show()