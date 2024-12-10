from data import Vector
import numpy as np
def cross_product(u, v):
    # a1, a2, a3 = u.data[0], u.data[1], u.data[2]
    # b1, b2, b3 = v.data[0], v.data[1], v.data[2]
    # c1 = a2*b3 - a3*b2
    # c2 = a3*b1 - a1*b3
    # c3 = a1*b2 - a2*b1
    # res = Vector([c1, c2, c3])
    # print(res)
    # res1 = np.cross(np.array(u.data), np.array(v.data))
    # print("np>",res1)
    return u.cross(v)