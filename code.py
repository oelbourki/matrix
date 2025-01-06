import math
class Color:
  RESET = '\033[0m'
  RED = '\033[91m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'

def assert_equal(actual, expected, test_name):
    if isinstance(actual, float) and isinstance(expected, float):
        if abs(actual - expected) < 1e-9:
            print(f"{Color.GREEN}PASS:{Color.RESET} {test_name}: Actual {actual} is close to expected {expected}")
            return True
        else:
            print(f"{Color.RED}FAIL:{Color.RESET} {test_name}: Actual {actual} is not equal to expected {expected}")
            return False
    elif type(actual) is type(expected) and actual == expected:
        print(f"{Color.GREEN}PASS:{Color.RESET} {test_name}: Actual {actual} is equal to expected {expected}")
        return True
    else:
        print(f"{Color.RED}FAIL:{Color.RESET} {test_name}: Actual {actual} is not equal to expected {expected}")
        return False
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
    
    return round(dot_product / (norm_u * norm_v),9)
class Vector:
    def __init__(self, data):
        self.data = list(data)

    def __str__(self):
      return str(self.data)

    def add(self, other):
      if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same length")
      for i in range(len(self.data)):
            self.data[i] += other.data[i]
      return self
    
    

    def sub(self, other):
      if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same length")
      for i in range(len(self.data)):
            self.data[i] -= other.data[i]
      return self
    def scl(self, scalar):
        for i in range(len(self.data)):
            self.data[i] *= scalar
        return self
    def dot(self, other):
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same length for dot product")
        return sum(self.data[i] * other.data[i] for i in range(len(self.data)))


    def norm_1(self):
        return sum(abs(x) for x in self.data)


    def norm(self):
        return math.sqrt(sum(x*x for x in self.data))


    def norm_inf(self):
        return max(abs(x) for x in self.data)
    def to_matrix(self,rows,cols):
        if rows*cols != len(self.data):
            raise ValueError(f"Cannot reshape vector of size {len(self.data)} into a matrix with {rows} rows and {cols} columns")
        matrix_data = []
        for i in range(rows):
            matrix_data.append(self.data[i*cols:(i+1)*cols])
        return Matrix(matrix_data)
    def __eq__(self, other):
        # Compare the values of the two vectors for equality
        if not isinstance(other, Vector):
            return False
        return self.data == other.data

class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data] # create a copy so that input data is immutable

    def __str__(self):
      return str(self.data)


    def add(self, other):
        if len(self.data) != len(other.data) or len(self.data[0]) != len(other.data[0]):
            raise ValueError("Matrices must have the same dimensions for addition")
        for i in range(len(self.data)):
          for j in range(len(self.data[0])):
            self.data[i][j] += other.data[i][j]
        return self

    def sub(self, other):
        if len(self.data) != len(other.data) or len(self.data[0]) != len(other.data[0]):
            raise ValueError("Matrices must have the same dimensions for subtraction")
        for i in range(len(self.data)):
          for j in range(len(self.data[0])):
            self.data[i][j] -= other.data[i][j]
        return self

    def scl(self, scalar):
        for i in range(len(self.data)):
          for j in range(len(self.data[0])):
            self.data[i][j] *= scalar
        return self
    def mul_vec(self, vec):
        if len(self.data[0]) != len(vec.data):
           raise ValueError("Matrix column size must match vector length for matrix-vector multiplication")
        result = [0 for _ in range(len(self.data))]
        for i in range(len(self.data)):
           for j in range(len(self.data[0])):
              result[i] += self.data[i][j] * vec.data[j]
        return Vector(result)

    def mul_mat(self, other):
      if len(self.data[0]) != len(other.data):
          raise ValueError("Number of columns in the first matrix must match number of rows in the second matrix")
      result = [[0 for _ in range(len(other.data[0]))] for _ in range(len(self.data))]
      for i in range(len(self.data)):
        for j in range(len(other.data[0])):
          for k in range(len(other.data)):
            result[i][j] += self.data[i][k] * other.data[k][j]
      return Matrix(result)


    def trace(self):
        if len(self.data) != len(self.data[0]):
            raise ValueError("Matrix must be square for trace")
        return sum(self.data[i][i] for i in range(len(self.data)))


    def transpose(self):
        rows = len(self.data)
        cols = len(self.data[0])
        transposed_data = [[0 for _ in range(rows)] for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                transposed_data[j][i] = self.data[i][j]
        return Matrix(transposed_data)



    def row_echelon(self):
        rows = len(self.data)
        cols = len(self.data[0])
        matrix_copy = [list(row) for row in self.data]
        lead = 0
        for r in range(rows):
            if lead >= cols:
                break
            i = r
            while abs(matrix_copy[i][lead]) < 1e-9:
                i += 1
                if i == rows:
                    i = r
                    lead += 1
                    if lead == cols:
                        break
            if lead == cols:
                break
            matrix_copy[r], matrix_copy[i] = matrix_copy[i], matrix_copy[r]
            lv = matrix_copy[r][lead]
            matrix_copy[r] = [mrx / float(lv) for mrx in matrix_copy[r]]
            for i in range(rows):
                if i != r:
                    lv = matrix_copy[i][lead]
                    matrix_copy[i] = [mrx - lv * matrix_copy[r][j] for j, mrx in enumerate(matrix_copy[i])]
            lead += 1
        for i in range(rows):
            for j in range(cols):
                matrix_copy[i][j] = round(matrix_copy[i][j], 15)
        return Matrix(matrix_copy)


    def determinant(self):
        n = len(self.data)
        if n == 1:
            return self.data[0][0]
        if n == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        if n == 3:
          return (
              self.data[0][0] * (self.data[1][1] * self.data[2][2] - self.data[1][2] * self.data[2][1])
              - self.data[0][1] * (self.data[1][0] * self.data[2][2] - self.data[1][2] * self.data[2][0])
              + self.data[0][2] * (self.data[1][0] * self.data[2][1] - self.data[1][1] * self.data[2][0])
          )
        if n == 4:
              det = 0
              for c in range(4):
                  submatrix = [row[:c] + row[c+1:] for row in self.data[1:]]
                  sign = 1 if c % 2 == 0 else -1
                  det += sign * self.data[0][c] * Matrix(submatrix).determinant()
              return det
        raise ValueError("Determinant is only implemented for matrices up to size 4x4")


    def inverse(self):
      n = len(self.data)
      if n != len(self.data[0]):
          raise ValueError("Matrix must be square for inverse")
      det = self.determinant()
      if abs(det) < 1e-9:
          raise ValueError("Matrix is singular, cannot calculate inverse")
      if n == 2:
          return Matrix([[self.data[1][1] / det, -self.data[0][1] / det], [-self.data[1][0] / det, self.data[0][0] / det]])
      else:
          # Calculate the adjugate matrix
          adj = [[0 for _ in range(n)] for _ in range(n)]
          for i in range(n):
              for j in range(n):
                  submatrix = [row[:j] + row[j+1:] for k, row in enumerate(self.data) if k != i]
                  sign = 1 if (i+j) % 2 == 0 else -1
                  adj[j][i] = sign * Matrix(submatrix).determinant()
          for i in range(n):
            for j in range(n):
              adj[i][j] /= det
          return Matrix(adj)


    def rank(self):
      rows = len(self.data)
      cols = len(self.data[0])
      reduced_matrix = self.row_echelon()
      rank = 0
      for row in reduced_matrix.data:
          if any(abs(x) > 1e-9 for x in row):
              rank += 1
      return rank

    def __eq__(self, other):
        # Compare the values of the two vectors for equality
        if not isinstance(other, Matrix):
            return False
        return self.data == other.data
def linear_combination(vectors, coefficients):
    if len(vectors) != len(coefficients):
      raise ValueError("Vectors and coefficients must have the same length")
    if not vectors:
      return Vector([])
    result = Vector([0 for _ in range(len(vectors[0].data))])
    for i, vec in enumerate(vectors):
        tmp = Vector(vec.data)
        tmp.scl(coefficients[i])
        result.add(tmp)
    return result



def lerp(u, v, t):
  if isinstance(u, (int, float)):
    return u + t * (v-u)
  elif isinstance(u, Vector):
    result = []
    for i in range(len(u.data)):
      result.append(u.data[i] + t * (v.data[i] - u.data[i]))
    return Vector(result)
  elif isinstance(u, Matrix):
      rows = len(u.data)
      cols = len(u.data[0])
      result = [[0. for _ in range(cols)] for _ in range(rows)]
      for i in range(rows):
         for j in range(cols):
            result[i][j] = u.data[i][j] + t * (v.data[i][j] - u.data[i][j])
      return Matrix(result)
  else:
    raise TypeError("Lerp is not supported for that type")

def cross_product(u, v):
  if len(u.data) != 3 or len(v.data) != 3:
    raise ValueError("Cross product is only defined for 3D vectors")
  result = [0, 0, 0]
  result[0] = u.data[1] * v.data[2] - u.data[2] * v.data[1]
  result[1] = u.data[2] * v.data[0] - u.data[0] * v.data[2]
  result[2] = u.data[0] * v.data[1] - u.data[1] * v.data[0]
  return Vector(result)


def projection(fov, ratio, near, far):
    tan_half_fov = math.tan(math.radians(fov / 2.0))
    m11 = 1.0 / (ratio * tan_half_fov)
    m22 = 1.0 / tan_half_fov
    m33 = -(far + near) / (far - near)
    m34 = -2.0 * far * near / (far - near)
    m43 = -1.0
    m44 = 0
    return Matrix([[m11, 0, 0, 0],
                [0, m22, 0, 0],
                [0, 0, m33, m34],
                [0, 0, m43, m44]])


# Test Execution with Verbose Output
def run_tests():
    test_results = []
    print(f"{Color.YELLOW}Running tests...{Color.RESET}\n")

    # Exercise 00 - Add, Subtract and Scale
    test_results.append(assert_equal(Vector([2., 3.]).add(Vector([5., 7.])),Vector([7.0, 10.0]), "Vector add"))

    test_results.append(assert_equal(Vector([2., 3.]).sub(Vector([5., 7.])), Vector([-3.0, -4.0]), "Vector sub"))

    test_results.append(assert_equal(Vector([2., 3.]).scl(2.), Vector([4.0, 6.0]), "Vector scl"))

    test_results.append(assert_equal(Matrix([[1., 2.], [3., 4.]]).add(Matrix([[7., 4.], [-2., 2.]])), Matrix([[8.0, 6.0], [1.0, 6.0]]),"Matrix add"))

    test_results.append(assert_equal(Matrix([[1., 2.], [3., 4.]]).sub(Matrix([[7., 4.], [-2., 2.]])), Matrix([[-6.0, -2.0], [5.0, 2.0]]),"Matrix sub"))

    test_results.append(assert_equal(Matrix([[1., 2.], [3., 4.]]).scl(2.), Matrix([[2.0, 4.0], [6.0, 8.0]]), "Matrix scl"))

    # Exercise 01 - Linear Combination
    e1 = Vector([1., 0., 0.])
    e2 = Vector([0., 1., 0.])
    e3 = Vector([0., 0., 1.])
    v1 = Vector([1., 2., 3.])
    v2 = Vector([0., 10., -100.])
    test_results.append(assert_equal(linear_combination([e1, e2, e3], [10., -2., 0.5]),Vector([10.0, -2.0, 0.5]),"Linear Combination 1"))
    test_results.append(assert_equal(linear_combination([v1, v2], [10., -2.]),Vector([10.0, 0.0, 230.0]),"Linear Combination 2"))

    # Exercise 02 - Linear Interpolation
    test_results.append(assert_equal(lerp(0., 1., 0.), 0.0, "Lerp 1"))
    test_results.append(assert_equal(lerp(0., 1., 1.), 1.0, "Lerp 2"))
    test_results.append(assert_equal(lerp(0., 1., 0.5), 0.5, "Lerp 3"))
    test_results.append(assert_equal(lerp(21., 42., 0.3), 27.3, "Lerp 4"))
    test_results.append(assert_equal(lerp(Vector([2., 1.]), Vector([4., 2.]), 0.3),Vector([2.6, 1.3]),"Lerp 5"))
    test_results.append(assert_equal(lerp(Matrix([[2., 1.], [3., 4.]]), Matrix([[20., 10.], [30., 40.]]), 0.5),Matrix([[11.0, 5.5], [16.5, 22.0]]),"Lerp 6"))

    # Exercise 03 - Dot Product
    u = Vector([0., 0.])
    v = Vector([1., 1.])
    test_results.append(assert_equal(u.dot(v), 0.0, "Dot Product 1"))

    u = Vector([1., 1.])
    v = Vector([1., 1.])
    test_results.append(assert_equal(u.dot(v), 2.0, "Dot Product 2"))

    u = Vector([-1., 6.])
    v = Vector([3., 2.])
    test_results.append(assert_equal(u.dot(v), 9.0, "Dot Product 3"))

    # Exercise 04 - Norm
    u = Vector([0., 0., 0.])
    test_results.append(assert_equal(u.norm_1(), 0.0, "Norm 1"))
    test_results.append(assert_equal(u.norm(), 0.0, "Norm 2"))
    test_results.append(assert_equal(u.norm_inf(), 0.0, "Norm 3"))


    u = Vector([1., 2., 3.])
    test_results.append(assert_equal(u.norm_1(), 6.0, "Norm 4"))
    test_results.append(assert_equal(u.norm(), 3.7416573867739413, "Norm 5"))
    test_results.append(assert_equal(u.norm_inf(), 3.0, "Norm 6"))

    u = Vector([-1., -2.])
    test_results.append(assert_equal(u.norm_1(), 3.0, "Norm 7"))
    test_results.append(assert_equal(u.norm(), 2.23606797749979, "Norm 8"))
    test_results.append(assert_equal(u.norm_inf(), 2.0, "Norm 9"))

    # Exercise 05 - Cosine
    u = Vector([1., 0.])
    v = Vector([1., 0.])
    test_results.append(assert_equal(angle_cos(u, v), 1.0, "Cosine 1"))

    u = Vector([1., 0.])
    v = Vector([0., 1.])
    test_results.append(assert_equal(angle_cos(u, v), 0.0, "Cosine 2"))

    u = Vector([-1., 1.])
    v = Vector([1., -1.])
    test_results.append(assert_equal(angle_cos(u, v), -1.0, "Cosine 3"))

    u = Vector([2., 1.])
    v = Vector([4., 2.])
    test_results.append(assert_equal(angle_cos(u, v), 1.0, "Cosine 4"))

    u = Vector([1., 2., 3.])
    v = Vector([4., 5., 6.])
    test_results.append(assert_equal(angle_cos(u, v), 0.9746318461970762, "Cosine 5"))

    # Exercise 06 - Cross Product
    u = Vector([0., 0., 1.])
    v = Vector([1., 0., 0.])
    test_results.append(assert_equal(cross_product(u, v),Vector([0.0, 1.0, 0.0]), "Cross Product 1"))

    u = Vector([1., 2., 3.])
    v = Vector([4., 5., 6.])
    test_results.append(assert_equal(cross_product(u, v), Vector([-3.0, 6.0, -3.0]), "Cross Product 2"))

    u = Vector([4., 2., -3.])
    v = Vector([-2., -5., 16.])
    test_results.append(assert_equal(cross_product(u, v),Vector([17.0, -58.0, -16.0]), "Cross Product 3"))

    # Exercise 07 - Linear Map, Matrix Multiplication
    u = Matrix([[1., 0.], [0., 1.]])
    v = Vector([4., 2.])
    test_results.append(assert_equal(u.mul_vec(v),Vector([4.0, 2.0]), "Matrix Mul Vec 1"))

    u = Matrix([[2., 0.], [0., 2.]])
    v = Vector([4., 2.])
    test_results.append(assert_equal(u.mul_vec(v),Vector([8.0, 4.0]), "Matrix Mul Vec 2"))

    u = Matrix([[2., -2.], [-2., 2.]])
    v = Vector([4., 2.])
    test_results.append(assert_equal(u.mul_vec(v),Vector([4.0, -4.0]), "Matrix Mul Vec 3"))

    u = Matrix([[1., 0.], [0., 1.]])
    v = Matrix([[1., 0.], [0., 1.]])
    test_results.append(assert_equal(u.mul_mat(v),Matrix([[1.0, 0.0], [0.0, 1.0]]), "Matrix Mul Mat 1"))

    u = Matrix([[1., 0.], [0., 1.]])
    v = Matrix([[2., 1.], [4., 2.]])
    test_results.append(assert_equal(u.mul_mat(v),Matrix([[2.0, 1.0], [4.0, 2.0]]),"Matrix Mul Mat 2"))

    u = Matrix([[3., -5.], [6., 8.]])
    v = Matrix([[2., 1.], [4., 2.]])
    test_results.append(assert_equal(u.mul_mat(v),Matrix([[-14.0, -7.0], [44.0, 22.0]]), "Matrix Mul Mat 3"))

    # Exercise 08 - Trace
    u = Matrix([[1., 0.], [0., 1.]])
    test_results.append(assert_equal(u.trace(), 2.0, "Trace 1"))

    u = Matrix([[2., -5., 0.], [4., 3., 7.], [-2., 3., 4.]])
    test_results.append(assert_equal(u.trace(), 9.0, "Trace 2"))

    u = Matrix([[-2., -8., 4.], [1., -23., 4.], [0., 6., 4.]])
    test_results.append(assert_equal(u.trace(), -21.0, "Trace 3"))


    # Exercise 09 - Transpose
    u = Matrix([[1, 2], [3, 4]])
    test_results.append(assert_equal(u.transpose(), Matrix([[1, 3], [2, 4]]),"Transpose 1"))

    # Exercise 10 - Row-Echelon Form
    u = Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    test_results.append(assert_equal(u.row_echelon(), Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),"Row Echelon 1"))

    u = Matrix([[1., 2.], [3., 4.]])
    test_results.append(assert_equal(u.row_echelon(), Matrix([[1.0, 0.0], [0.0, 1.0]]), "Row Echelon 2"))

    u = Matrix([[1., 2.], [2., 4.]])
    test_results.append(assert_equal(u.row_echelon(),Matrix([[1.0, 2.0], [0.0, 0.0]]), "Row Echelon 3"))

    u = Matrix([[8., 5., -2., 4., 28.], [4., 2.5, 20., 4., -4.], [8., 5., 1., 4., 17.]])
    test_results.append(assert_equal(u.row_echelon(), Matrix([[1, 0.625, 0, 0, -12.1666666666667],[0,     0, 1, 0, -3.66666666666667],[0,     0, 0, 1,              29.5]]), "Row Echelon 4"))

    # Exercise 11 - Determinant
    u = Matrix([[1., -1.], [-1., 1.]])
    test_results.append(assert_equal(u.determinant(), 0.0, "Determinant 1"))

    u = Matrix([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
    test_results.append(assert_equal(u.determinant(), 8.0, "Determinant 2"))

    u = Matrix([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.]])
    test_results.append(assert_equal(u.determinant(), -174.0, "Determinant 3"))

    u = Matrix([[8., 5., -2., 4.], [4., 2.5, 20., 4.], [8., 5., 1., 4.], [28., -4., 17., 1.]])
    test_results.append(assert_equal(u.determinant(), 1032.0, "Determinant 4"))


    # Exercise 12 - Inverse
    u = Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    test_results.append(assert_equal(u.inverse(), Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "Inverse 1"))

    u = Matrix([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
    test_results.append(assert_equal(u.inverse(), Matrix([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]), "Inverse 2"))

    u = Matrix([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.]])
    test_results.append(assert_equal(u.inverse(), Matrix([[0.6494252873563218, 0.09770114942528735, -0.6551724137931034], [-0.7816091954022989, -0.1264367816091954, 0.9655172413793104], [0.14367816091954023, 0.07471264367816091, -0.20689655172413793]]), "Inverse 3"))

    # Exercise 13 - Rank
    u = Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    test_results.append(assert_equal(u.rank(), 3, "Rank 1"))

    u = Matrix([[1., 2., 0., 0.], [2., 4., 0., 0.], [-1., 2., 1., 1.]])
    test_results.append(assert_equal(u.rank(), 2, "Rank 2"))

    u = Matrix([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.], [21., 18., 7.]])
    test_results.append(assert_equal(u.rank(), 3, "Rank 3"))


    #Exercise 14 - projection matrix
    test_results.append(assert_equal(projection(90,1,0.1,100), Matrix([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, -1.002002002002002, -0.2002002002002002], [0, 0, -1.0, 0]] )  ,"projection matrix 1" ) )


    # Test Summary
    print(f"\n{Color.YELLOW}Test Summary:{Color.RESET}")
    total_tests = len(test_results)
    passed_tests = sum(test_results)
    failed_tests = total_tests - passed_tests
    print(f"Total Tests: {total_tests}")
    print(f"{Color.GREEN}Passed Tests: {passed_tests}{Color.RESET}")
    print(f"{Color.RED}Failed Tests: {failed_tests}{Color.RESET}")
    if failed_tests > 0:
        print(f"{Color.RED}Tests failed{Color.RESET}")
        exit(1)
    else:
        print(f"{Color.GREEN}All tests passed{Color.RESET}")


# Entry point
if __name__ == "__main__":
    run_tests()

