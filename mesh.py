from dolfin import *
import matplotlib.pyplot as plt

plt.rcParams.update({
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.dpi": 100 #change resolution, standard is 100
        })

mesh = RectangleMesh(Point(0,0), Point(25, 25), 10, 10)

num_points = mesh.num_vertices()

print(f"The mesh has {num_points} number of points.")
# Plot mesh
plot(mesh)
plt.xlabel("x")
plt.ylabel("y")
plt.show()