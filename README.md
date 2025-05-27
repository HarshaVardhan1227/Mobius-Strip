# Mobius-Strip

# Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simpson

# Define the MobiusStrip Class
class MobiusStrip:

# Initialize the Mobius strip parameters.
def init(self,R,w,n):
self.R = R
self.w = w
self.n = n
self.u_range = np.linspace(0, 2*np.pi, n)
self.v_range = np.linspace(-w/2, w/2, n)
self.points = None
self.generate_mesh()

# Generate the 3D Mesh
def generate_mesh(self):
u, v = np.meshgrid(self.u_range, self.v_range)
x = (self.R + v * np.cos(u/2)) * np.cos(u)
y = (self.R + v * np.cos(u/2)) * np.sin(u)
z = v * np.sin(u/2)

self.points = np.stack((x, y, z), axis=-1)
Compute Surface Area
def cal_surface_area(self):

u, v = np.meshgrid(self.u_range, self.v_range)

dx_du = - (self.R + v * np.cos(u/2)) * np.sin(u) - 0.5 * v * np.sin(u/2) * np.cos(u)
dy_du = (self.R + v * np.cos(u/2)) * np.cos(u) - 0.5 * v * np.sin(u/2) * np.sin(u)
dz_du = 0.5 * v * np.cos(u/2)

dx_dv = np.cos(u/2) * np.cos(u)
dy_dv = np.cos(u/2) * np.sin(u)
dz_dv = np.sin(u/2)

cross_x = dy_du * dz_dv - dz_du * dy_dv
cross_y = dz_du * dx_dv - dx_du * dz_dv
cross_z = dx_du * dy_dv - dy_du * dx_dv

magnitude = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)

area = simpson(simpson(magnitude, self.v_range), self.u_range)

return area
Compute Edge Length
def cal_edge_length(self):
u = self.u_range
v_top = self.w/2 * np.ones_like(u)
v_bottom = -self.w/2 * np.ones_like(u)

x_top = (self.R + v_top * np.cos(u/2)) * np.cos(u)
y_top = (self.R + v_top * np.cos(u/2)) * np.sin(u)
z_top = v_top * np.sin(u/2)

x_bottom = (self.R + v_bottom * np.cos(u/2)) * np.cos(u)
y_bottom = (self.R + v_bottom * np.cos(u/2)) * np.sin(u)
z_bottom = v_bottom * np.sin(u/2)

dx_top = np.gradient(x_top, u)
dy_top = np.gradient(y_top, u)
dz_top = np.gradient(z_top, u)

dx_bottom = np.gradient(x_bottom, u)
dy_bottom = np.gradient(y_bottom, u)
dz_bottom = np.gradient(z_bottom, u)

ds_top = np.sqrt(dx_top**2 + dy_top**2 + dz_top**2)
ds_bottom = np.sqrt(dx_bottom**2 + dy_bottom**2 + dz_bottom**2)

length = simpson(ds_top, u)

return length
# Visualize the Mobius Strip
def plot(self):

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = self.points[:, :, 0]
y = self.points[:, :, 1]
z = self.points[:, :, 2]

ax.plot_surface(x, y, z, color='black', alpha=0.8, rstride=2, cstride=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Mobius Strip (R={self.R}, w={self.w})')

plt.tight_layout()
plt.show()
# Main Function
if name == "main":
R=float(input())
W=float(input())
N=int(input())
mobius = MobiusStrip(R,W,N)
surface_area = mobius.cal_surface_area()
edge_length = mobius.cal_edge_length()
print(f"Surface Area of strip: {surface_area:.4f}")
print(f"Edge Length of strip: {edge_length:.4f}")
mobius.plot()

# Summary
Import Libraries → numpy, matplotlib, scipy.integrate.
Initialize Class → Set parameters (R, w, n).
Generate Mesh → Compute (x, y, z) points.
Compute Surface Area → Using partial derivatives & integration.
Compute Edge Length → Using arc length integration.
Visualize → 3D plot of the Mobius strip.
Run Program → Compute properties & display results.
