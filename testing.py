import RayTracingObjects as rto
import numpy as np
from PIL import Image as im
from Scene import Scene
from Camera import Camera


def main():
    # set up the camera
    eye_pos = np.array([0, 0, 0]).reshape([3, ])
    eye_to = np.array([0, 0, 1]).reshape([3, ])
    cam = Camera(eye_pos, eye_to, x_res=128, y_res=128)

    p = np.array([0, -1, 0])
    norm = -p
    plane = rto.Plane(p, norm, light_source=True, light_color=np.array([0, .2, .2]), light_strength=1.0)
    # plane = rto.Plane(p, norm, color=[1, 1, 1])

    # sphere1 = rto.Sphere(.25, np.array([0, .25, 1]), color=np.array([1, 0, 0]))
    sphere1 = rto.Sphere(1, np.array([0, 0, 2]), color=np.array([.537, .812, .941]))
    # sphere2 = rto.Sphere(.5, np.array([0, -1, 1]), color=np.array([0, 0, 0]))
    # sphere3 = rto.Sphere(5, np.array([0, 0, 7.5]), color=np.array([1, 0, 0]))

    # sphere1 = rto.Sphere(1, np.array([0, 0, 2]),
    #                      light_source=True, light_color=np.array([1, 0, 0]), light_strength=1.0)

    # scene = Scene(cam, plane, sphere1, sphere2, sphere3, color=[0, 0, 0])
    # scene = Scene(cam, plane, sphere1, color=[0, 0, 0])
    scene = Scene(cam, plane, sphere1, color=[1, 1, 1])
    # scene = Scene(cam, plane, sphere1, color=[.537, .812, .941])
    i = scene.render(n_rays=50, n_bounces=2)
    im.fromarray(np.uint8(i)).show()


main()
