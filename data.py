import math

class Vector:
    def __init__(self, data):
        if not all(isinstance(x, (int, float)) for x in data):
            raise TypeError("All elements in the vector must be integers or floats.")
        self.data = [float(x) for x in data]

    def size(self):
        return len(self.data)

    def reshape(self, rows, cols):
        if self.size() != rows * cols:
            raise ValueError("Cannot reshape vector into the desired dimensions.")
        reshaped_data = [self.data[i * cols:(i + 1) * cols] for i in range(rows)]
        return Matrix(reshaped_data)

    def add(self, other):
        if self.size() != other.size():
            raise ValueError("Vectors must be of the same size to add.")
        return Vector([x + y for x, y in zip(self.data, other.data)])

    def subtract(self, other):
        if self.size() != other.size():
            raise ValueError("Vectors must be of the same size to subtract.")
        return Vector([x - y for x, y in zip(self.data, other.data)])

    def dot(self, other):
        if self.size() != other.size():
            raise ValueError("Vectors must be of the same size for dot product.")
        return sum(x * y for x, y in zip(self.data, other.data))

    def scale(self, scalar):
        return Vector([x * scalar for x in self.data])
    def norm_1(self):
        """
        Compute the 1-norm (Manhattan norm) of the vector.
        
        Returns:
        float: The 1-norm of the vector.
        """
        return sum(abs(x) for x in self.data)

    def norm(self):
        """
        Compute the 2-norm (Euclidean norm) of the vector.
        
        Returns:
        float: The 2-norm of the vector.
        """
        return math.sqrt(sum(x**2 for x in self.data))

    def norm_inf(self):
        """
        Compute the infinity norm (supremum norm) of the vector.
        
        Returns:
        float: The infinity norm of the vector.
        """
        return max(abs(x) for x in self.data)
    # def cos()
    def __str__(self):
        return f"Vector: {self.data}"
    def cross(self, other):
        # Ensure both vectors are 3-dimensional
        if len(self.data) != 3 or len(other.data) != 3:
            raise ValueError("Cross product is only defined for 3-dimensional vectors.")
        
        x1, y1, z1 = self.data
        x2, y2, z2 = other.data
        
        # Calculate the cross product
        return Vector([y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2])

    def __eq__(self, other):
        # Compare the values of the two vectors for equality
        if not isinstance(other, Vector):
            return False
        return self.data == other.data


class Matrix:
    def __init__(self, data):
        if not all(isinstance(row, list) for row in data):
            raise TypeError("Matrix must be initialized with a 2D list.")
        if not all(isinstance(x, (int, float)) for row in data for x in row):
            raise TypeError("All elements in the matrix must be integers or floats.")
        if not all(len(row) == len(data[0]) for row in data):
            raise ValueError("All rows in the matrix must have the same number of columns.")
        self.data = [[float(x) for x in row] for row in data]

    def shape(self):
        rows = len(self.data)
        cols = len(self.data[0]) if rows > 0 else 0
        return rows, cols

    def is_square(self):
        rows, cols = self.shape()
        return rows == cols

    def trace(self):
        if not self.is_square():
            raise ValueError("Matrix must be square to compute the trace.")
        return sum(self.data[i][i] for i in range(len(self.data)))

    def transpose(self):
        transposed_data = [[self.data[j][i] for j in range(len(self.data))] for i in range(len(self.data[0]))]
        return Matrix(transposed_data)

    def add(self, other):
        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same shape to add.")
        rows, cols = self.shape()
        return Matrix([[self.data[i][j] + other.data[i][j] for j in range(cols)] for i in range(rows)])

    def subtract(self, other):
        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same shape to subtract.")
        rows, cols = self.shape()
        return Matrix([[self.data[i][j] - other.data[i][j] for j in range(cols)] for i in range(rows)])

    def multiply(self, other):
        if self.shape()[1] != other.shape()[0]:
            raise ValueError("Number of columns in first matrix must equal number of rows in second matrix.")
        rows, cols = self.shape()[0], other.shape()[1]
        common_dim = self.shape()[1]
        result_data = [[sum(self.data[i][k] * other.data[k][j] for k in range(common_dim)) for j in range(cols)]
                       for i in range(rows)]
        return Matrix(result_data)
    def mul_vec(self, other):
        if isinstance(other, Vector):
            tmp = Matrix([other.data])
            # print("self:", self.shape())
            # print("tmp:", tmp.shape())
            result = tmp.multiply(self)
            # print(result.data)
            return Vector(result.data[0])
    def mul_mat(self, other):
        return self.multiply(other)
    def scale(self, scalar):
        rows, cols = self.shape()
        return Matrix([[self.data[i][j] * scalar for j in range(cols)] for i in range(rows)])

    def __str__(self):
        return "Matrix:\n" + "\n".join(["\t" + str(row) for row in self.data])
    def __eq__(self, other):
        # Compare the values of the two vectors for equality
        if not isinstance(other, Matrix):
            return False
        return self.data == other.data


# Testing the extended classes

if __name__ == "__main__":
    # Vectors
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    print(v1.add(v2))        # Vector addition
    print(v1.subtract(v2))   # Vector subtraction
    print(v1.dot(v2))        # Dot product
    print(v1.scale(2))       # Scaling

    # Matrices
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    m3 = Matrix([[1, 2, 3], [4, 5, 6]])
    m4 = Matrix([[7, 8], [9, 10], [11, 12]])

    print(m1.add(m2))        # Matrix addition
    print(m1.subtract(m2))   # Matrix subtraction
    print("m1.shape: ",m1.shape())
    print("m4.shape: ",m4.shape())

    print(m1.multiply(m4))   # Matrix multiplication
    print(m3.multiply(m4))   # Matrix multiplication
    print(m1.scale(3))       # Scaling
    print(m1.transpose())    # Transpose
    print(m1.trace())        # Trace of a square matrix
