__author__ = 'eribeiro'


#
import numpy as np

#
# Ray in 3-D space
# Parametric representation of a line, p(t) = e + (s - e)*t.
#
class Ray:
    # Properties
    e = np.zeros((3, 1))  # Starting point
    s = np.zeros((3, 1))  # End point

    # Methods
    'Constructor'

    def __init__(self, e, s):
        self.e = e
        self.s = s


    def get3DPoint(self, t):
        p = self.e + (self.s - self.e) * t  # p(t) = e + (s - e) * t
        return p


#
# Sphere
#
#
class Sphere:
    # Properties
    Center = np.zeros((3, 1))  # 3-D coordinates of center
    Radius = 0  # Radius
    Color = np.zeros((3, 1))  # RGB components

    # Methods
    'Constructor'

    def __init__(self, c, r, k):
        # c: center 3x1 array
        # r: scalar double/float
        #  c: color 3x1 array (R,G,B)
        self.Center = c
        self.Radius = r
        self.Color = k

    'Ray-sphere intersection. It returns the smallest t. '
    def Intersect(self, ray):
        # ray: ray object  p(t) = e + (s - e) * t

        # For calculations, I prefer to use the notation
        # similar to the one in the slides or associated paper.
        d = ray.s - ray.e  # Direction of the ray
        e = ray.e          # Ray's starting point

        c = self.Center  # Sphere center
        r = self.Radius  # Sphere radius

        # Check whether the ray intersects the sphere
        A = np.dot(d, d)
        B = 2.0 * np.dot(d, (e - c))
        C = np.dot((e - c), (e - c)) - r * r

        #delta = B*B - A * C
        delta = B*B - 4.0 * A * C
        if delta < 0:
            return float("inf")         # Ray didn't intersect sphere
        else:
            # Compute value of parameter t at the intersection point
            t1 = (-B - np.sqrt(delta)) / (2.0 * A)
            t2 = (-B + np.sqrt(delta)) / (2.0 * A)

            # We are interested only in the closest intersection.
            # We take the smaller t
            t = np.min([t1, t2])
            return t


#
# My simple test program...
#

# Create a sphere of radius r = 30 centered at c = (0,0,-100).
Center = np.array((0, 0, -100.0)).transpose()
Radius = 30.0
Color = np.array((255, 0, 0)).transpose()
S = Sphere(Center, Radius, Color)

# Set focal distance
f = 50.0

# Create a ray
e = np.array((0.0, 0.0, 0.0)).transpose()
s = np.array((8.0, -8.0, -f)).transpose()
R = Ray(e, s)

# Intersect the ray with the sphere
t = S.Intersect(R)



if float('inf') == t:
    print ("Ray does not intersect the sphere.")
else:
    print ("This ray intersects the sphere at t = ", t)
    print ("The 3-D point is p(t) = ", R.get3DPoint(t))



