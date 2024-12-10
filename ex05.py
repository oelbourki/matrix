import math
from data import Vector

def angle_cos(u, v):
    """
    Compute the cosine of the angle between two vectors.
    
    Args:
    u (Vector): The first vector.
    v (Vector): The second vector.
    
    Returns:
    float: The cosine of the angle between the two vectors.
    """
    if len(u.data) != len(v.data):
        raise ValueError("Vectors must have the same dimension.")
    
    dot_product = u.dot(v)
    norm_u = u.norm()
    norm_v = v.norm()
    
    if norm_u == 0 or norm_v == 0:
        raise ValueError("Undefined behavior for zero vectors.")
    
    return round(dot_product / (norm_u * norm_v),16)


# Testing the function
if __name__ == "__main__":
    # Test cases
    u1 = Vector([1.0, 0.0])
    v1 = Vector([1.0, 0.0])
    print(f"Cosine: {angle_cos(u1, v1)}")  # Expected: 1.0

    u2 = Vector([1.0, 0.0])
    v2 = Vector([0.0, 1.0])
    print(f"Cosine: {angle_cos(u2, v2)}")  # Expected: 0.0

    u3 = Vector([-1.0, 1.0])
    v3 = Vector([1.0, -1.0])
    print(f"Cosine: {angle_cos(u3, v3)}")  # Expected: -1.0

    u4 = Vector([2.0, 1.0])
    v4 = Vector([4.0, 2.0])
    print(f"Cosine: {angle_cos(u4, v4)}")  # Expected: 1.0

    u5 = Vector([1.0, 2.0, 3.0])
    v5 = Vector([4.0, 5.0, 6.0])
    print(f"Cosine: {angle_cos(u5, v5)}")  # Expected: ~0.974631846
