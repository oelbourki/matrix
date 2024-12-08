from data import Vector

def linear_combination(vectors, coefs):
    if len(vectors) != len(coefs):
        raise ValueError("Number of vectors and coefficients must match.")
    if not vectors:
        raise ValueError("No vectors provided.")
    
    result = vectors[0].scale(coefs[0])
    for i in range(1, len(vectors)):
        result = result.add(vectors[i].scale(coefs[i]))
    return result

# Testing the function
if __name__ == "__main__":
    e1 = Vector([1.0, 0.0, 0.0])
    e2 = Vector([0.0, 1.0, 0.0])
    e3 = Vector([0.0, 0.0, 1.0])
    v1 = Vector([1.0, 2.0, 3.0])
    v2 = Vector([0.0, 10.0, -100.0])
    
    # Test 1: Linear combination of basis vectors
    result1 = linear_combination([e1, e2, e3], [10.0, -2.0, 0.5])
    print("Result 1:", result1)  # Expected: [10.0, -2.0, 0.5]
    
    # Test 2: Linear combination of arbitrary vectors
    result2 = linear_combination([v1, v2], [10.0, -2.0])
    print("Result 2:", result2)  # Expected: [10.0, 0.0, 230.0]