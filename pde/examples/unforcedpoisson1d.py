import pde
from matplotlib import pyplot as plt
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

f,a = plt.subplots(3)
prob = pde.problems.instpoisson1d.unforced
prob.plot(a)

solver = pde.solvers.linearsolver(prob)
solver.solve()
solver.plot(a)

plt.show()