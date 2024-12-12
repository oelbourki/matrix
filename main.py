from ex02 import lerp
from data import Vector, Matrix
from ex01 import linear_combination
from ex05 import angle_cos
from ex06 import cross_product
def test_vector_operations():
    print("Running Vector Tests")
    u = Vector([2.0, 3.0])
    v = Vector([5.0, 7.0])

    # Test Vector addition
    u_add = u.add(v)
    assert u_add.data == [7.0, 10.0], f"Addition failed: {u_add.data}"

    # Test Vector subtraction
    u = Vector([2.0, 3.0])  # Reset u
    u_sub = u.subtract(v)
    assert u_sub.data == [-3.0, -4.0], f"Subtraction failed: {u_sub.data}"

    # Test Vector scaling
    u = Vector([2.0, 3.0])  # Reset u
    u_scaled = u.scale(2.0)
    assert u_scaled.data == [4.0, 6.0], f"Scaling failed: {u_scaled.data}"

    print("Vector tests passed!\n")


def test_matrix_operations():
    print("Running Matrix Tests")
    u = Matrix([[1.0, 2.0], [3.0, 4.0]])
    v = Matrix([[7.0, 4.0], [-2.0, 2.0]])

    # Test Matrix addition
    u_add = u.add(v)
    assert u_add.data == [[8.0, 6.0], [1.0, 6.0]], f"Addition failed: {u_add.data}"

    # Test Matrix subtraction
    u = Matrix([[1.0, 2.0], [3.0, 4.0]])  # Reset u
    u_sub = u.subtract(v)
    assert u_sub.data == [[-6.0, -2.0], [5.0, 2.0]], f"Subtraction failed: {u_sub.data}"

    # Test Matrix scaling
    u = Matrix([[1.0, 2.0], [3.0, 4.0]])  # Reset u
    u_scaled = u.scale(2.0)
    assert u_scaled.data == [[2.0, 4.0], [6.0, 8.0]], f"Scaling failed: {u_scaled.data}"

    print("Matrix tests passed!")
# Test linear_combination function
def test_linear_combination():
    e1 = Vector([1., 0., 0.])
    e2 = Vector([0., 1., 0.])
    e3 = Vector([0., 0., 1.])
    v1 = Vector([1., 2., 3.])
    v2 = Vector([0., 10., -100.])

    # Linear combination of e1, e2, e3 with coefficients [10, -2, 0.5]
    result = linear_combination([e1, e2, e3], [10., -2., 0.5])
    expected = Vector([10., -2., 0.5])
    assert result.data == expected.data, f"Expected {expected.data}, but got {result.data}"

    # Linear combination of v1, v2 with coefficients [10, -2]
    result = linear_combination([v1, v2], [10., -2.])
    expected = Vector([10., 0., 230.])
    assert result.data == expected.data, f"Expected {expected.data}, but got {result.data}"

    print("All tests passed!")



# Test lerp function
def test_lerp():
    # Scalar lerp tests
    assert lerp(0., 1., 0.) == 0.0, "Test 1 failed"
    assert lerp(0., 1., 1.) == 1.0, "Test 2 failed"
    assert lerp(0., 1., 0.5) == 0.5, "Test 3 failed"
    assert lerp(21., 42., 0.3) == 27.3, f"Test 4 failed {lerp(21., 42., 0.3)}"

    # Vector lerp tests
    v1 = Vector([2., 1.])
    v2 = Vector([4., 2.])
    result = lerp(v1, v2, 0.3)
    expected = Vector([2.6, 1.3])
    assert result.data == expected.data, f"Test 5 failed. Expected {expected.data}, but got {result.data}"

    # Matrix lerp tests
    m1 = Matrix([[2., 1.], [3., 4.]])
    m2 = Matrix([[20., 10.], [30., 40.]])
    result = lerp(m1, m2, 0.5)
    expected = Matrix([[11., 5.5], [16.5, 22.]])
    assert result.data == expected.data, f"Test 6 failed. Expected {expected.data}, but got {result.data}"

    print("All tests passed!")
def test_dot():
    # Test 1: Dot product of [0., 0.] and [1., 1.]
    u = Vector([0., 0.])
    v = Vector([1., 1.])
    assert u.dot(v) == 0.0, f"Expected 0.0 but got {u.dot(v)}"
    
    # Test 2: Dot product of [1., 1.] and [1., 1.]
    u = Vector([1., 1.])
    v = Vector([1., 1.])
    assert u.dot(v) == 2.0, f"Expected 2.0 but got {u.dot(v)}"
    
    # Test 3: Dot product of [-1., 6.] and [3., 2.]
    u = Vector([-1., 6.])
    v = Vector([3., 2.])
    assert u.dot(v) == 9.0, f"Expected 9.0 but got {u.dot(v)}"

def test_norms():
    # Test 1: Norms of [0., 0., 0.]
    u = Vector([0., 0., 0.])
    assert u.norm_1() == 0.0, f"Expected 0.0 but got {u.norm_1()}"
    assert u.norm() == 0.0, f"Expected 0.0 but got {u.norm()}"
    assert u.norm_inf() == 0.0, f"Expected 0.0 but got {u.norm_inf()}"

    # Test 2: Norms of [1., 2., 3.]
    u = Vector([1., 2., 3.])
    assert u.norm_1() == 6.0, f"Expected 6.0 but got {u.norm_1()}"
    assert u.norm() == 3.7416573867739413, f"Expected 3.7416573867739413 but got {u.norm()}"
    assert u.norm_inf() == 3.0, f"Expected 3.0 but got {u.norm_inf()}"

    # Test 3: Norms of [-1., -2.]
    u = Vector([-1., -2.])
    assert u.norm_1() == 3.0, f"Expected 3.0 but got {u.norm_1()}"
    assert u.norm() == 2.23606797749979, f"Expected 2.23606797749979 but got {u.norm()}"
    assert u.norm_inf() == 2.0, f"Expected 2.0 but got {u.norm_inf()}"
def test_angle_cos():
    # Test 1: Angle cosine between [1., 0.] and [1., 0.]
    u = Vector([1., 0.])
    v = Vector([1., 0.])
    assert angle_cos(u, v) == 1.0, f"Expected 1.0 but got {angle_cos(u, v)}"
    
    # Test 2: Angle cosine between [1., 0.] and [0., 1.]
    u = Vector([1., 0.])
    v = Vector([0., 1.])
    assert angle_cos(u, v) == 0.0, f"Expected 0.0 but got {angle_cos(u, v)}"
    
    # Test 3: Angle cosine between [-1., 1.] and [1., -1.]
    u = Vector([-1., 1.])
    v = Vector([1., -1.])
    assert angle_cos(u, v) == -1.0, f"Expected -1.0 but got {angle_cos(u, v)}"
    
    # Test 4: Angle cosine between [2., 1.] and [4., 2.]
    u = Vector([2., 1.])
    v = Vector([4., 2.])
    assert angle_cos(u, v) == 1.0, f"Expected 1.0 but got {angle_cos(u, v)}"
    
    # Test 5: Angle cosine between [1., 2., 3.] and [4., 5., 6.]
    u = Vector([1., 2., 3.])
    v = Vector([4., 5., 6.])
    assert angle_cos(u, v) == 0.9746318461970762, f"Expected 0.9746318461970762 but got {angle_cos(u, v)}"

def test_cross_product():
    # Test 1: Cross product of [0., 0., 1.] and [1., 0., 0.]
    u = Vector([0., 0., 1.])
    v = Vector([1., 0., 0.])
    result = cross_product(u, v)
    print("results", result)
    assert result == Vector([0., 1., 0.]), f"Test 1 Failed: {result}"

    # Test 2: Cross product of [1., 2., 3.] and [4., 5., 6.]
    u = Vector([1., 2., 3.])
    v = Vector([4., 5., 6.])
    result = cross_product(u, v)
    assert result == Vector([-3., 6., -3.]), f"Test 2 Failed: {result}"

    # Test 3: Cross product of [4., 2., -3.] and [-2., -5., 16.]
    u = Vector([4., 2., -3.])
    v = Vector([-2., -5., 16.])
    result = cross_product(u, v)
    assert result == Vector([17., -58., -16.]), f"Test 3 Failed: {result}"

    print("All tests passed!")

def test_matrix_transpose():
    # Test 1: 2x2 Identity Matrix
    u = Matrix([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    u_transposed = u.transpose()
    expected = Matrix([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    assert u_transposed == expected, f"Test 1 failed: {u_transposed.data}"

    # Test 2: 2x3 Matrix
    u = Matrix([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    u_transposed = u.transpose()
    expected = Matrix([
        [1.0, 4.0],
        [2.0, 5.0],
        [3.0, 6.0],
    ])
    assert u_transposed == expected, f"Test 2 failed: {u_transposed.data}"

    # Test 3: 3x3 Matrix
    u = Matrix([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    u_transposed = u.transpose()
    expected = Matrix([
        [1.0, 4.0, 7.0],
        [2.0, 5.0, 8.0],
        [3.0, 6.0, 9.0],
    ])
    assert u_transposed == expected, f"Test 3 failed: {u_transposed.data}"

    print("All transpose tests passed!")

def test_matrix_trace():
    # Test 1
    u = Matrix([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    assert u.trace() == 2.0, f"Test 1 failed: {u.trace()}"

    # Test 2
    u = Matrix([
        [2.0, -5.0, 0.0],
        [4.0, 3.0, 7.0],
        [-2.0, 3.0, 4.0],
    ])
    assert u.trace() == 9.0, f"Test 2 failed: {u.trace()}"

    # Test 3
    u = Matrix([
        [-2.0, -8.0, 4.0],
        [1.0, -23.0, 4.0],
        [0.0, 6.0, 4.0],
    ])
    assert u.trace() == -21.0, f"Test 3 failed: {u.trace()}"

    print("All tests passed!")
  

# Run the tests

# test_vector_operations()
# test_matrix_operations()
# test_linear_combination()
# test_lerp()
# test_dot()
# test_norms()
# test_angle_cos()
# test_cross_product()
test_matrix_transpose()
test_matrix_trace()