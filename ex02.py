from data import Vector, Matrix

def round_v(r):
    if isinstance(r, list):
        data = []
        for x in r:
            data.append(round(x, 1))
        return data
    else:
        return round(r, 1)
def lerp(u, v, t):

    if isinstance(u, float) and isinstance(v, float):
        return round(u * (1.0 - t) + v * t, 1)
        # return a * (1.0 - f) + (b * f)

    if not (0 <= t <= 1):
        raise ValueError("t must be between 0 and 1.")
    result = u.scale(1 - t).add(v.scale(t))
    # for i,r in enumerate(result.data):
    #     for c in r
    res = [round_v(r) for r in result.data]
    # print(res)
    result.data = res
    # print(round(result.data))
    return result


# Testing the function
if __name__ == "__main__":
    # Vectors
    v1 = Vector([2.0, 1.0])
    v2 = Vector([4.0, 2.0])
    
    print("Lerp between vectors:")
    print(lerp(v1, v2, 0.0))  # Expected: [2.0, 1.0]
    print(lerp(v1, v2, 1.0))  # Expected: [4.0, 2.0]
    print(lerp(v1, v2, 0.5))  # Expected: [3.0, 1.5]
    print(lerp(v1, v2, 0.3))  # Expected: [2.6, 1.3]

    # Matrices
    m1 = Matrix([[2.0, 1.0], [3.0, 4.0]])
    m2 = Matrix([[20.0, 10.0], [30.0, 40.0]])
    
    print("\nLerp between matrices:")
    print(lerp(m1, m2, 0.0))  # Expected: [[2.0, 1.0], [3.0, 4.0]]
    print(lerp(m1, m2, 1.0))  # Expected: [[20.0, 10.0], [30.0, 40.0]]
    print(lerp(m1, m2, 0.5))  # Expected: [[11.0, 5.5], [16.5, 22.0]]
    print(lerp(m1, m2, 0.3))  # Expected: [[7.4, 4.3], [11.1, 15.8]]
