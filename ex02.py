from data import Vector, Matrix

def lerp(u, v, t):
    """
    Linear interpolation between two objects of the same type.
    
    Args:
    u (Vector or Matrix): The starting point.
    v (Vector or Matrix): The ending point.
    t (float): The interpolation factor (0 <= t <= 1).
    
    Returns:
    Vector or Matrix: The interpolated object.
    """
    if not (0 <= t <= 1):
        raise ValueError("t must be between 0 and 1.")
    return u.scale(1 - t).add(v.scale(t))


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
