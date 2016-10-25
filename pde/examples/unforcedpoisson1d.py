import pde
from matplotlib import pyplot as plt

f,a = plt.subplots(3)
prob = pde.problems.instpoisson1d.unforced
prob.plot(a)

plt.show()