import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Given points A, B, C
A = np.array([0, 4, 1])
B = np.array([-10, 0, 8])
C = np.array([9, -3, 6])

# Calculate midpoints
M_AB = (A + B) / 2
M_BC = (B + C) / 2

# Calculate slopes of perpendicular bisectors
S_AB = -1 / ((B[1] - A[1]) / (B[0] - A[0]))
S_BC = -1 / ((C[1] - B[1]) / (C[0] - B[0]))

# Calculate circumcircle center
x_o = (S_AB * M_AB[0] - S_BC * M_BC[0] + M_BC[1] - M_AB[1]) / (S_AB - S_BC)
y_o = S_AB * (x_o - M_AB[0]) + M_AB[1]
z_o = M_AB[2]  # Assuming the points are coplanar

# Calculate circumcircle radius
r = np.linalg.norm(A - np.array([x_o, y_o, z_o]))

# Generate points on the circular arc
t = np.linspace(0, 2 * np.pi, 100)
arc_points = np.array([x_o + r * np.cos(t), y_o + r * np.sin(t), np.full_like(t, z_o)]).T

# Plot the circular arc
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*zip(A, B, C), c='red', marker='o', label='Given Points')
ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], label='Circular Arc')
ax.scatter(x_o, y_o, z_o, c='blue', marker='x', label='Circumcircle Center')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
plt.show()
