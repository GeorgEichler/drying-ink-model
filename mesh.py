from dolfin import *
import matplotlib.pyplot as plt

mesh = RectangleMesh(Point(0,0), Point(25, 25), 10, 10)


# Plot mesh
plot(mesh)
plt.xlabel("x")
plt.ylabel("y")
plt.show()