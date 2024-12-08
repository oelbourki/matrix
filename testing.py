import numpy as np
from data import Vector, Matrix
# Vectors
v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])

# Vector Addition
try:
    print("Vector Addition:", v1.add(v2).data)
    np_result = np.add(np.array(v1.data), np.array(v2.data))
    print("NumPy Vector Addition:", np_result.tolist())
except Exception as e:
    print("Vector Addition Error:", e)

# Vector Subtraction
try:
    print("Vector Subtraction:", v1.subtract(v2).data)
    np_result = np.subtract(np.array(v1.data), np.array(v2.data))
    print("NumPy Vector Subtraction:", np_result.tolist())
except Exception as e:
    print("Vector Subtraction Error:", e)

# Dot Product
try:
    print("Dot Product:", v1.dot(v2))
    np_result = np.dot(np.array(v1.data), np.array(v2.data))
    print("NumPy Dot Product:", np_result)
except Exception as e:
    print("Dot Product Error:", e)

# Scaling
try:
    print("Scaling Vector:", v1.scale(2).data)
    np_result = np.multiply(np.array(v1.data), 2)
    print("NumPy Scaling:", np_result.tolist())
except Exception as e:
    print("Scaling Vector Error:", e)

# Matrices
m1 = Matrix([[1, 2], [3, 4]])
m2 = Matrix([[5, 6], [7, 8]])
m3 = Matrix([[1, 2, 3], [4, 5, 6]])
m4 = Matrix([[7, 8], [9, 10], [11, 12]])

# Matrix Addition
try:
    print("Matrix Addition:", m1.add(m2).data)
    np_result = np.add(np.array(m1.data), np.array(m2.data))
    print("NumPy Matrix Addition:", np_result.tolist())
except Exception as e:
    print("Matrix Addition Error:", e)

# Matrix Subtraction
try:
    print("Matrix Subtraction:", m1.subtract(m2).data)
    np_result = np.subtract(np.array(m1.data), np.array(m2.data))
    print("NumPy Matrix Subtraction:", np_result.tolist())
except Exception as e:
    print("Matrix Subtraction Error:", e)

# Matrix Shape
print("Matrix Shape m1:", m1.shape())
print("Matrix Shape m4:", m4.shape())

# Matrix Multiplication
try:
    print("Matrix Multiplication m1 * m4:", m1.multiply(m4).data)
    np_result = np.dot(np.array(m1.data), np.array(m4.data))
    print("NumPy Matrix Multiplication m1 * m4:", np_result.tolist())
except Exception as e:
    print("Matrix Multiplication Error m1 * m4:", e)

try:
    print("Matrix Multiplication m3 * m4:", m3.multiply(m4).data)
    np_result = np.dot(np.array(m3.data), np.array(m4.data))
    print("NumPy Matrix Multiplication m3 * m4:", np_result.tolist())
except Exception as e:
    print("Matrix Multiplication Error m3 * m4:", e)

# Scaling
try:
    print("Scaling Matrix m1:", m1.scale(3).data)
    np_result = np.multiply(np.array(m1.data), 3)
    print("NumPy Scaling Matrix m1:", np_result.tolist())
except Exception as e:
    print("Scaling Matrix Error:", e)

# Transpose
try:
    print("Transpose m1:", m1.transpose().data)
    np_result = np.transpose(np.array(m1.data))
    print("NumPy Transpose m1:", np_result.tolist())
except Exception as e:
    print("Transpose Error:", e)

# Trace
try:
    print("Trace m1:", m1.trace())
    np_result = np.trace(np.array(m1.data))
    print("NumPy Trace m1:", np_result)
except Exception as e:
    print("Trace Error:", e)

# Invalid Cases
try:
    print("Invalid Vector Addition (size mismatch):", v1.add(Vector([1, 2])))
except Exception as e:
    print("Expected Error for Vector Addition:", e)

try:
    print("Invalid Matrix Addition (shape mismatch):", m1.add(m4))
except Exception as e:
    print("Expected Error for Matrix Addition:", e)

try:
    print(m1.shape())
    print(m2.shape())
    
    print("valid Matrix Multiplication:", m1.multiply(m2))
except Exception as e:
    print("Expected Error for Matrix Multiplication:", e)
