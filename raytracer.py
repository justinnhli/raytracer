#!/usr/bin/env python3

# pylint: disable = too-many-lines

from math import sqrt, isclose, sin, cos, tan, pi as PI
from pathlib import Path
from typing import Any, Optional, Union, Generator


EPSILON = 0.00001


class Matrix: # pylint: disable = too-many-public-methods
    """A matrix.

    >>> Point(3, -2, 5) + Vector(-2, 3, 1)
    Point(1, 1, 6)

    >>> Point(3, 2, 1) - Point(5, 6, 7)
    Vector(-2, -4, -6)

    >>> Point(3, 2, 1) - Vector(5, 6, 7)
    Point(-2, -4, -6)

    >>> Vector(3, 2, 1) - Vector(5, 6, 7)
    Vector(-2, -4, -6)

    >>> Tuple4(1, -2, 3, -4) * 3.5 == Tuple4(3.5, -7, 10.5, -14)
    True

    >>> 0.5 * Tuple4(1, -2, 3, -4) == Tuple4(0.5, -1, 1.5, -2)
    True

    >>> Tuple4(1, -2, 3, -4) / 2 == Tuple4(0.5, -1, 1.5, -2)
    True

    >>> Vector(1, 0, 0).magnitude == 1
    True

    >>> Vector(0, 1, 0).magnitude == 1
    True

    >>> Vector(0, 0, 1).magnitude == 1
    True

    >>> Vector(1, 2, 3).magnitude == sqrt(14)
    True

    >>> Vector(-1, -2, -3).magnitude == sqrt(14)
    True

    >>> Vector(4, 0, 0).normalize() == Vector(1, 0, 0)
    True

    >>> Vector(1, 2, 3).normalize().magnitude == 1
    True

    >>> Vector(1, 2, 3).dot(Vector(2, 3, 4))
    20

    >>> Vector(1, 2, 3).cross(Vector(2, 3, 4))
    Vector(-1, 2, -1)

    >>> Vector(2, 3, 4).cross(Vector(1, 2, 3))
    Vector(1, -2, 1)

    >>> m1 = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]])
    >>> m2 = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]])
    >>> m1 == m2
    True

    >>> m1 = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]])
    >>> m2 = Matrix([[2, 3, 4, 5], [6, 7, 8, 9], [8, 7, 6, 5], [4, 3, 2, 1]])
    >>> m1 == m2
    False

    >>> m1 = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]])
    >>> m2 = Matrix([[-2, 1, 2, 3], [3, 2, 1, -1], [4, 3, 6, 5], [1, 2, 7, 8]])
    >>> m3 = Matrix([[20, 22, 50, 48], [44, 54, 114, 108], [40, 58, 110, 102], [16, 26, 46, 42]])
    >>> m1 @ m2 == m3
    True

    >>> m1 = Matrix([[1, 2, 3, 4], [2, 4, 4, 2], [8, 6, 4, 1], [0, 0, 0, 1]])
    >>> t = Tuple4(1, 2, 3, 1)
    >>> m1 @ t
    Point(18, 24, 33)

    >>> m1 = Matrix([[0, 1, 2, 4], [1, 2, 4, 8], [2, 4, 8, 16], [4, 8, 16, 32]])
    >>> m2 = identity()
    >>> m1 @ m2 == m1
    True

    >>> m1 = Matrix([[0, 9, 3, 0], [9, 8, 0, 8], [1, 8, 5, 3], [0, 0, 5, 8]])
    >>> m2 = Matrix([[0, 9, 1, 0], [9, 8, 8, 0], [3, 0, 5, 5], [0, 8, 3, 8]])
    >>> m1.transpose() == m2
    True
    >>> m1.transpose().transpose() == m1
    True

    >>> identity().transpose() == identity()
    True

    >>> Matrix([[1, 5], [-3, 2]]).determinant
    17

    >>> Matrix([[1, 5, 0], [-3, 2, 7], [0, 6, -3]]).submatrix(0, 2)
    Matrix([[-3, 2], [0, 6]])

    >>> Matrix([[-6, 1, 1, 6], [-8, 5, 8, 6], [-1, 0, 8, 2], [-7, 1, -1, 1]]).submatrix(2, 1)
    Matrix([[-6, 1, 6], [-8, 8, 6], [-7, -1, 1]])

    >>> Matrix([[3, 5, 0], [2, -1, -7], [6, -1, 5]]).minor(1, 0)
    25

    >>> m = Matrix([[3, 5, 0], [2, -1, -7], [6, -1, 5]])
    >>> m.cofactor(0, 0)
    -12
    >>> m.cofactor(1, 0)
    -25

    >>> Matrix([[1, 2, 6], [-5, 8, -4], [2, 6, 4]]).determinant
    -196

    >>> Matrix([[-2, -8, 3, 5], [-3, 1, 7, 3], [1, 2, -9, 6], [-6, 7, 7, -9]]).determinant
    -4071

    >>> m1 = Matrix([[3, -9, 7, 3], [3, -8, 2, -9], [-4, 4, 4, 1], [-6, 5, -1, 1]])
    >>> m2 = Matrix([[8, 2, 2, 2], [3, -1, 7, 0], [7, 0, 5, 4], [6, -2, 0, 5]])
    >>> m1 @ m2 @ m2.inverse() == m1
    True
    """

    def __init__(self, values):
        # type: (list[list[float]]) -> None
        self.rows = values
        self.height = len(values)
        self.width = len(values[0])
        self._cols = None # type: list[list[float]]
        self._inverse = None # type: Optional[Matrix]

    @property
    def cols(self):
        # type: () -> list[list[float]]
        if self._cols is None:
            self._cols = [
                [self.rows[r][c] for r in range(self.height)]
                for c in range(self.width)
            ]
        return self._cols

    @property
    def is_tuple(self):
        # type: () -> bool
        return self.height == 1 and self.width == 4

    @property
    def is_vector(self):
        # type: () -> bool
        return self.is_tuple and self.rows[0][3] == 0

    @property
    def is_point(self):
        # type: () -> bool
        return self.is_tuple and self.rows[0][3] == 1

    @property
    def x(self):
        # type: () -> float
        return self.rows[0][0]

    @property
    def y(self):
        # type: () -> float
        return self.rows[0][1]

    @property
    def z(self):
        # type: () -> float
        return self.rows[0][2]

    @property
    def w(self): # pylint: disable = invalid-name
        # type: () -> float
        return self.rows[0][3]

    @property
    def magnitude(self):
        # type: () -> float
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def __eq__(self, other):
        # type: (Any) -> bool
        return (
            self.height == other.height and self.width == other.width
            and all(
                isclose(self_val, other_val, abs_tol=EPSILON)
                for self_row, other_row in zip(self.rows, other.rows)
                for self_val, other_val in zip(self_row, other_row)
            )
        )

    def __str__(self):
        # type: () -> str
        return repr(self)

    def __repr__(self):
        # type: () -> str
        if self.is_tuple:
            vals = [str(abs(i) if i == 0 else i) for i in self.rows[0]]
            if self.is_vector:
                return f'Vector({", ".join(vals[:-1])})'
            elif self.is_point:
                return f'Point({", ".join(vals[:-1])})'
            else:
                return f'Tuple4({", ".join(vals)})'
        else:
            return f'Matrix({str(self.rows)})'

    def __add__(self, other):
        # type: (Matrix) -> Matrix
        return Matrix([
            [val1 + val2 for val1, val2 in zip(row1, row2)]
            for row1, row2 in zip(self.rows, other.rows)
        ])

    def __sub__(self, other):
        # type: (Matrix) -> Matrix
        return Matrix([
            [val1 - val2 for val1, val2 in zip(row1, row2)]
            for row1, row2 in zip(self.rows, other.rows)
        ])

    def __neg__(self):
        # type: () -> Matrix
        return Matrix([
            [-val for val in row]
            for row in self.rows
        ])

    def __mul__(self, other):
        # type: (Union[int,float]) -> Matrix
        result = []
        for row in self.rows:
            result.append([val * other for val in row])
        return Matrix(result)

    def __rmul__(self, other):
        # type: (Union[int,float]) -> Matrix
        return self * other

    def __truediv__(self, other):
        # type: (Union[int,float]) -> Matrix
        return self * (1 / other)

    def __matmul__(self, other):
        # type: (Matrix) -> Matrix
        is_tuple = False
        if other.is_tuple:
            other = other.transpose()
            is_tuple = True
        result = []
        for r in range(self.height):
            row = self.rows[r]
            result_row = []
            for c in range(other.width):
                col = other.cols[c]
                result_row.append(sum(a * b for a, b in zip(row, col)))
            result.append(result_row)
        if is_tuple:
            return Matrix(result).transpose()
        else:
            return Matrix(result)

    def normalize(self):
        # type: () -> Matrix
        magnitude = self.magnitude
        return Vector(self.x / magnitude, self.y / magnitude, self.z / magnitude)

    def reflect(self, other):
        # type: (Matrix) -> Matrix
        normal = other.normalize()
        return self - normal * 2 * self.dot(normal)

    def dot(self, other):
        # type: (Matrix) -> float
        return (self @ other.transpose()).rows[0][0]

    def cross(self, other):
        # type: (Matrix) -> Matrix
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def transpose(self):
        # type: () -> Matrix
        return Matrix(self.cols)

    @property
    def determinant(self):
        # type: () -> float
        if self.height == 2 and self.width == 2:
            return self.rows[0][0] * self.rows[1][1] - self.rows[0][1] * self.rows[1][0]
        else:
            return sum(self.rows[0][i] * self.cofactor(0, i) for i in range(self.width))

    def submatrix(self, dr, dc): # pylint: disable = invalid-name
        # type: (int, int) -> Matrix
        result = []
        for r, row in enumerate(self.rows):
            if r == dr:
                continue
            result.append(row[:dc] + row[dc + 1:])
        return Matrix(result)

    def minor(self, dr, dc): # pylint: disable = invalid-name
        # type: (int, int) -> float
        return self.submatrix(dr, dc).determinant

    def cofactor(self, dr, dc): # pylint: disable = invalid-name
        # type: (int, int) -> float
        if (dr + dc) % 2 == 0:
            return self.minor(dr, dc)
        else:
            return -self.minor(dr, dc)

    @property
    def invertible(self):
        # type: () -> bool
        return self.determinant != 0

    def inverse(self):
        # type: () -> Matrix
        if self._inverse is None:
            result = []
            for r in range(self.height):
                result.append([self.cofactor(r, c) for c in range(self.width)])
            self._inverse = Matrix(result).transpose() / self.determinant
        return self._inverse

    def translate(self, x, y, z):
        # type: (float, float, float) -> Matrix
        """
        >>> identity().translate(5, -3, 2) @ Point(-3, 4, 5) == Point(2, 1, 7)
        True

        >>> identity().translate(5, -3, 2).inverse() @ Point(-3, 4, 5) == Point(-8, 7, 3)
        True

        >>> identity().translate(5, -3, 2) @ Vector(-3, 4, 5) == Vector(-3, 4, 5)
        True
        """
        return (
            Matrix([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
            @ self
        )

    def scale(self, x, y, z):
        # type: (float, float, float) -> Matrix
        """
        >>> identity().scale(2, 3, 4) @ Point(-4, 6, 8) == Point(-8, 18, 32)
        True

        >>> identity().scale(2, 3, 4) @ Vector(-4, 6, 8) == Vector(-8, 18, 32)
        True

        >>> identity().scale(2, 3, 4).inverse() @ Point(-4, 6, 8) == Point(-2, 2, 2)
        True
        """
        return (
            Matrix([[x, 0, 0, 0], [0, y, 0, 0], [0, 0, z, 0], [0, 0, 0, 1]])
            @ self
        )

    def rotate_x(self, r):
        # type: (float) -> Matrix
        return (
            Matrix([[1, 0, 0, 0], [0, cos(r), -sin(r), 0], [0, sin(r), cos(r), 0], [0, 0, 0, 1]])
            @ self
        )

    def rotate_y(self, r):
        # type: (float) -> Matrix
        return (
            Matrix([[cos(r), 0, sin(r), 0], [0, 1, 0, 0], [-sin(r), 0, cos(r), 0], [0, 0, 0, 1]])
            @ self
        )

    def rotate_z(self, r):
        # type: (float) -> Matrix
        return (
            Matrix([[cos(r), -sin(r), 0, 0], [sin(r), cos(r), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            @ self
        )

    def shear(self, x_y, x_z, y_x, y_z, z_x, z_y):
        # type: (float, float, float, float, float, float) -> Matrix
        """
        >>> identity().shear(1, 0, 0, 0, 0, 0) @ Point(2, 3, 4) == Point(5, 3, 4)
        True
        >>> identity().shear(0, 1, 0, 0, 0, 0) @ Point(2, 3, 4) == Point(6, 3, 4)
        True
        >>> identity().shear(0, 0, 1, 0, 0, 0) @ Point(2, 3, 4) == Point(2, 5, 4)
        True
        >>> identity().shear(0, 0, 0, 1, 0, 0) @ Point(2, 3, 4) == Point(2, 7, 4)
        True
        >>> identity().shear(0, 0, 0, 0, 1, 0) @ Point(2, 3, 4) == Point(2, 3, 6)
        True
        >>> identity().shear(0, 0, 0, 0, 0, 1) @ Point(2, 3, 4) == Point(2, 3, 7)
        True
        """
        return (
            Matrix([[1, x_y, x_z, 0], [y_x, 1, y_z, 0], [z_x, z_y, 1, 0], [0, 0, 0, 1]])
            @ self
        )


def Tuple4(x, y, z, w): # pylint: disable = invalid-name
    # type: (float, float, float, float) -> Matrix
    return Matrix([[x, y, z, w]])


def Vector(x, y, z): # pylint: disable = invalid-name
    # type: (float, float, float) -> Matrix
    return Matrix([[x, y, z, 0]])


def Point(x, y, z): # pylint: disable = invalid-name
    # type: (float, float, float) -> Matrix
    return Matrix([[x, y, z, 1]])


def identity():
    # type: () -> Matrix
    """
    >>> identity() == Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    True
    """
    return Matrix([(i * [0.0]) + [1.0] + (3 - i) * [0.0] for i in range(4)])


class Color:
    """An RGB color, specified between 0 and 1.

    >>> Color(0.9, 0.6, 0.75) + Color(0.7, 0.1, 0.25) == Color(1.6, 0.7, 1)
    True

    >>> Color(0.9, 0.6, 0.75) - Color(0.7, 0.1, 0.25) == Color(0.2, 0.5, 0.5)
    True

    >>> Color(0.2, 0.3, 0.4) * 2 == Color(0.4, 0.6, 0.8)
    True

    >>> 2 * Color(0.2, 0.3, 0.4) == Color(0.4, 0.6, 0.8)
    True

    >>> Color(1, 0.2, 0.4) * Color(0.9, 1, 0.1) == Color(0.9, 0.2, 0.04)
    True
    """

    def __init__(self, red, green, blue):
        # type: (float, float, float) -> None
        self.vals = [red, green, blue]

    @property
    def red(self):
        # type: () -> float
        return self.vals[0]

    @property
    def green(self):
        # type: () -> float
        return self.vals[1]

    @property
    def blue(self):
        # type: () -> float
        return self.vals[2]

    def __eq__(self, other):
        # type: (Any) -> bool
        return all(isclose(a, b, abs_tol=EPSILON) for a, b in zip(self, other))

    def __iter__(self):
        # type: () -> Generator[float, None, None]
        yield from self.vals

    def __str__(self):
        # type: () -> str
        return repr(self)

    def __repr__(self):
        # type: () -> str
        return f'Color({self.red}, {self.green}, {self.blue})'

    def __add__(self, other):
        # type: (Color) -> Color
        return Color(*(a + b for a, b in zip(self, other)))

    def __sub__(self, other):
        # type: (Color) -> Color
        return Color(*(a - b for a, b in zip(self, other)))

    def __mul__(self, other):
        # type: (Union[float, Color]) -> Color
        if isinstance(other, Color):
            return Color(
                self.red * other.red,
                self.green * other.green,
                self.blue * other.blue,
            )
        else:
            return Color(*(other * a for a in self.vals))

    def __rmul__(self, other):
        # type: (float) -> Color
        return self.__mul__(other)


class Canvas:
    """
    >>> canvas = Canvas(5, 3)
    >>> canvas.set_pixel(0, 0, Color(1.5, 0, 0))
    >>> canvas.set_pixel(1, 2, Color(0, 0.5, 0))
    >>> canvas.set_pixel(2, 4, Color(-0.5, 0, 1))
    >>> print(canvas.ppm)
    P3
    5 3
    255
    255 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 128 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 255
    <BLANKLINE>
    """

    def __init__(self, width, height):
        # type: (int, int) -> None
        self.width = width
        self.height = height
        self.canvas = {}
        for r in range(self.height):
            for c in range(self.width):
                self.canvas[(r, c)] = Color(0, 0, 0)

    def set_pixel(self, row, col, color):
        # type: (float, float, Color) -> None
        key = (round(row), round(col))
        if key in self.canvas:
            self.canvas[key] = color

    def get_pixel(self, row, col):
        # type: (int, int) -> Color
        return self.canvas[(row, col)]

    @property
    def ppm(self):
        # type: () -> str
        lines = []
        lines.append('P3')
        lines.append(f'{self.width} {self.height}')
        lines.append('255')
        for r in range(self.height):
            row = [] # type: list[str]
            for c in range(self.width):
                pixel = self.canvas[(r, c)]
                row.extend(str(min(max(round(i * 255), 0), 255)) for i in pixel.vals)
            lines.append(' '.join(row))
        lines.append('')
        return '\n'.join(lines)

    def save(self, path):
        # type: (Path) -> None
        with path.open('w') as fd:
            fd.write(self.ppm)


class Ray:

    def __init__(self, origin, direction):
        # type: (Matrix, Matrix) -> None
        self.origin = origin
        self.direction = direction

    def __eq__(self, other):
        # type: (Any) -> bool
        return self.origin == other.origin and self.direction == other.direction

    def __repr__(self):
        # type: () -> str
        return f'Ray({self.origin}, {self.direction})'

    def position(self, t): # pylint: disable = invalid-name
        # type: (float) -> Matrix
        """
        >>> ray = Ray(Point(2, 3, 4), Vector(1, 0, 0))
        >>> ray.position(0) == Point(2, 3, 4)
        True
        >>> ray.position(1) == Point(3, 3, 4)
        True
        >>> ray.position(-1) == Point(1, 3, 4)
        True
        >>> ray.position(2.5) == Point(4.5, 3, 4)
        True
        """
        return self.origin + t * self.direction

    def transform(self, transform):
        # type: (Matrix) -> Ray
        """
        >>> actual = Ray(Point(1, 2, 3), Vector(0, 1, 0)).transform(identity().translate(3, 4, 5))
        >>> expected = Ray(Point(4, 6, 8), Vector(0, 1, 0))
        >>> actual == expected
        True
        >>> actual = Ray(Point(1, 2, 3), Vector(0, 1, 0)).transform(identity().scale(2, 3, 4))
        >>> expected = Ray(Point(2, 6, 12), Vector(0, 3, 0))
        >>> actual == expected
        True
        """
        origin = transform @ self.origin
        direction = transform @ self.direction
        return Ray(origin, direction)


class Computation: # FIXME delete this when reflections are done

    def __init__(self, t, shape, point, eye_vector, normal_vector): # pylint: disable = invalid-name
        # type: (float, Shape, Matrix, Matrix, Matrix) -> None
        self.t = t # pylint: disable = invalid-name
        self.shape = shape
        self.point = point
        self.eye_vector = eye_vector
        self._normal_vector = normal_vector

    @property
    def inside(self):
        # type: () -> bool
        return self._normal_vector.dot(self.eye_vector) < 0

    @property
    def normal_vector(self):
        # type: () -> Matrix
        if self.inside:
            return -self._normal_vector
        else:
            return self._normal_vector

    def __repr__(self):
        # type: () -> str
        return f'Computation({self.t}, {self.shape}, {self.point}, {self.eye_vector}, {self.normal_vector})'


class Intersection:

    def __init__(self, t, shape, ray): # pylint: disable = invalid-name
        # type: (float, Shape, Ray) -> None
        """
        >>> intersection = Intersection(4, Sphere(), Ray(Point(0, 0, -5), Vector(0, 0, 1)))
        >>> intersection.point == Point(0, 0, -1)
        True
        >>> intersection.eye_vector == Vector(0, 0, -1)
        True
        >>> intersection.normal_vector == Vector(0, 0, -1)
        True

        >>> intersection = Intersection(1, Sphere(), Ray(Point(0, 0, 0), Vector(0, 0, 1)))
        >>> intersection.point == Point(0, 0, 1)
        True
        >>> intersection.eye_vector == Vector(0, 0, -1)
        True
        >>> intersection.normal_vector == Vector(0, 0, -1)
        True
        >>> intersection.inside
        True
        """
        self.t = t # pylint: disable = invalid-name
        self.shape = shape
        self.ray = ray

    def __repr__(self):
        # type: () -> str
        return f'Intersection({self.t}, {self.shape}, {self.ray})'

    @property
    def point(self):
        # type: () -> Matrix
        return self.ray.position(self.t)

    @property
    def eye_vector(self):
        # type: () -> Matrix
        return -self.ray.direction

    @property
    def _normal_vector(self):
        # type: () -> Matrix
        return self.shape.normal(self.point)

    @property
    def inside(self):
        # type: () -> bool
        return self._normal_vector.dot(self.eye_vector) < 0

    @property
    def normal_vector(self):
        # type: () -> Matrix
        if self.inside:
            return -self._normal_vector
        else:
            return self._normal_vector

    @property
    def over_point(self):
        # type: () -> Matrix
        return self.point + EPSILON / 2 * self.normal_vector

    def prep(self, ray): # FIXME delete this when reflections are done
        # type: (Ray) -> Computation
        point = ray.position(self.t)
        return Computation(
            self.t,
            self.shape,
            point,
            -ray.direction,
            self.shape.normal(point),
        )


class Shape:

    def __init__(self, transform=None, material=None):
        # type: (Matrix, Material) -> None
        if transform is None:
            self.transform = identity()
        else:
            self.transform = transform
        if material is None:
            self.material = Material()
        else:
            self.material = material

    def __repr__(self):
        # type: () -> str
        return f'{type(self)}({self.transform})' # FIXME ignores transform and material

    def intersect(self, ray):
        # type: (Ray) -> list[Intersection]
        raise NotImplementedError()

    def normal(self, point):
        # type: (Matrix) -> Matrix
        raise NotImplementedError()


class Sphere(Shape):

    def intersect(self, ray): # pylint: disable = invalid-name
        # type: (Ray) -> list[Intersection]
        """
        >>> [intersection.t for intersection in Sphere().intersect(Ray(Point(0, 0, -5), Vector(0, 0, 1)))]
        [4.0, 6.0]

        >>> [intersection.t for intersection in Sphere().intersect(Ray(Point(0, 1, -5), Vector(0, 0, 1)))]
        [5.0, 5.0]

        >>> [intersection.t for intersection in Sphere().intersect(Ray(Point(0, 2, -5), Vector(0, 0, 1)))]
        []

        >>> [intersection.t for intersection in Sphere().intersect(Ray(Point(0, 0, 0), Vector(0, 0, 1)))]
        [-1.0, 1.0]

        >>> [intersection.t for intersection in Sphere().intersect(Ray(Point(0, 0, 5), Vector(0, 0, 1)))]
        [-6.0, -4.0]

        >>> intersections = Sphere(identity().scale(2, 2, 2)).intersect(Ray(Point(0, 0, -5), Vector(0, 0, 1)))
        >>> [intersection.t for intersection in intersections]
        [3.0, 7.0]

        >>> intersections = Sphere(identity().translate(5, 0, 0)).intersect(Ray(Point(0, 0, -5), Vector(0, 0, 1)))
        >>> [intersection.t for intersection in intersections]
        []
        """
        new_ray = ray.transform(self.transform.inverse())
        vector = new_ray.origin - Point(0, 0, 0)
        a = new_ray.direction.dot(new_ray.direction) # pylint: disable = invalid-name
        b = 2 * new_ray.direction.dot(vector) # pylint: disable = invalid-name
        c = vector.dot(vector) - 1 # pylint: disable = invalid-name
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return []
        else:
            return [
                Intersection((-b - sqrt(discriminant)) / (2 * a), self, ray),
                Intersection((-b + sqrt(discriminant)) / (2 * a), self, ray),
            ]

    def normal(self, point):
        # type: (Matrix) -> Matrix
        """
        >>> Sphere().normal(Point(1, 0, 0)) == Vector(1, 0, 0)
        True

        >>> Sphere().normal(Point(0, 1, 0)) == Vector(0, 1, 0)
        True

        >>> Sphere().normal(Point(0, 0, 1)) == Vector(0, 0, 1)
        True
        """
        object_normal = ((self.transform.inverse() @ point) - Point(0, 0, 0)).normalize()
        return (self.transform.inverse().transpose() @ object_normal).normalize()


class PointLight:

    def __init__(self, position, intensity):
        # type: (Matrix, Color) -> None
        self.position = position
        self.intensity = intensity


class Material:

    def __init__(self, color=None, ambient=0.1, diffuse=0.9, specular=0.9, shininess=200):
        # type: (Color, float, float, float, float) -> None
        if color is None:
            self.color = Color(1, 1, 1)
        else:
            self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess


def hit(*intersections):
    # type: (*Intersection) -> Intersection
    """
    >>> hit(Intersection(1, None, None), Intersection(2, None, None))
    Intersection(1, None, None)

    >>> hit(Intersection(1, None, None), Intersection(-1, None, None))
    Intersection(1, None, None)

    >>> hit(Intersection(-1, None, None), Intersection(-2, None, None))

    >>> hit(
    ... Intersection(5, None, None),
    ... Intersection(7, None, None),
    ... Intersection(-3, None, None),
    ... Intersection(2, None, None),
    ... )
    Intersection(2, None, None)
    """
    if not intersections:
        return None
    result = None
    for intersection in intersections:
        if intersection.t >= 0 and (result is None or intersection.t < result.t):
            result = intersection
    return result


def lighting(point, light, eye_vector, normal_vector, material=None, in_shadow=False):
    # type: (Matrix, PointLight, Matrix, Matrix, Material, bool) -> Color
    """
    >>> actual = lighting(
    ...     Point(0, 0, 0),
    ...     PointLight(Point(0, 0, -10), Color(1, 1, 1)),
    ...     Vector(0, 0, -1),
    ...     Vector(0, 0, -1)
    ... )
    >>> expected = Color(1.9, 1.9, 1.9)
    >>> actual == expected
    True

    >>> actual = lighting(
    ...     Point(0, 0, 0),
    ...     PointLight(Point(0, 0, -10), Color(1, 1, 1)),
    ...     Vector(0, 1, -1),
    ...     Vector(0, 0, -1),
    ... )
    >>> expected = Color(1, 1, 1)
    >>> actual == expected
    True

    >>> actual = lighting(
    ...     Point(0, 0, 0),
    ...     PointLight(Point(0, 10, -10),
    ...     Color(1, 1, 1)),
    ...     Vector(0, 0, -1),
    ...     Vector(0, 0, -1),
    ... )
    >>> expected = Color(0.7364, 0.7364, 0.7364)
    >>> actual == expected
    True

    >>> actual = lighting(
    ...     Point(0, 0, 0),
    ...     PointLight(Point(0, 10, -10),
    ...     Color(1, 1, 1)),
    ...     Vector(0, -1, -1),
    ...     Vector(0, 0, -1),
    ... )
    >>> expected = Color(1.63640, 1.63640, 1.63640)
    >>> actual == expected
    True

    >>> actual = lighting(
    ...     Point(0, 0, 0),
    ...     PointLight(Point(0, 0, 10), Color(1, 1, 1)),
    ...     Vector(0, 0, -1),
    ...     Vector(0, 0, -1),
    ... )
    >>> expected = Color(0.1, 0.1, 0.1)
    >>> actual == expected
    True

    >>> actual = lighting(
    ...     Point(0, 0, 0),
    ...     PointLight(Point(0, 0, -10), Color(1, 1, 1)),
    ...     Vector(0, 0, -1),
    ...     Vector(0, 0, -1),
    ...     in_shadow=True,
    ... )
    >>> expected = Color(0.1, 0.1, 0.1)
    >>> actual == expected
    True
    """
    if material is None:
        material = Material()
    eye_vector = eye_vector.normalize()
    normal_vector = normal_vector.normalize()
    # combine the surface color with the light's color/intensity
    effective_color = material.color * light.intensity
    # find the direction of the light source
    light_vector = (light.position - point).normalize()
    # calculate the ambient component
    ambient = effective_color * material.ambient
    # calculate the angle between the light and the normal vector
    # a negative value means that the light is on the other side of the surface
    light_dot_normal = light_vector.dot(normal_vector)
    if in_shadow or light_dot_normal < 0:
        diffuse = Color(0, 0, 0)
        specular = Color(0, 0, 0)
    else:
        # calculate the diffuse component
        diffuse = effective_color * material.diffuse * light_dot_normal
        # calculate the angle between the camera and the normal vector
        # a negative value means that the camera is on the other side of the surface
        reflect_dot_eye = (-light_vector).reflect(normal_vector).dot(eye_vector)
        if reflect_dot_eye < 0:
            specular = Color(0, 0, 0)
        else:
            # calculate the specular component
            specular = light.intensity * material.specular * reflect_dot_eye ** material.shininess
    # return the combined contributions of the components
    return ambient + diffuse + specular


class World:

    def __init__(self, light=None, shapes=None):
        # type: (PointLight, list[Shape]) -> None
        if light is None:
            light = PointLight(Point(-10, 10, -10), Color(1, 1, 1))
        self.light = light
        self.shapes = [] # type: list[Shape]
        if shapes is None:
            shapes = [
                Sphere(material=Material(
                    color=Color(0.8, 1.0, 0.6),
                    diffuse=0.7,
                    specular=0.2,
                )),
                Sphere(identity().scale(0.5, 0.5, 0.5)),
            ]
        self.shapes = list(shapes)

    def intersect(self, ray):
        # type: (Ray) -> list[Intersection]
        """
        >>> intersections = World().intersect(Ray(Point(0, 0, -5), Vector(0, 0, 1)))
        >>> [intersection.t for intersection in intersections]
        [4.0, 4.5, 5.5, 6.0]
        """
        results = []
        for shape in self.shapes:
            results.extend(shape.intersect(ray))
        return sorted(results, key=(lambda intersection: intersection.t))

    def shade_hit(self, intersection):
        # type: (Intersection) -> Color
        """
        >>> world = World()
        >>> ray = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        >>> intersection = hit(*world.shapes[0].intersect(ray))
        >>> world.shade_hit(intersection) == Color(0.380665, 0.47583, 0.28550)
        True

        >>> world = World(light=PointLight(Point(0, 0.25, 0), Color(1, 1, 1)))
        >>> ray = Ray(Point(0, 0, 0), Vector(0, 0, 1))
        >>> intersection = hit(*world.shapes[1].intersect(ray))
        >>> world.shade_hit(intersection) == Color(0.90498, 0.90498, 0.90498)
        True

        >>> world = World(
        ...     light=PointLight(Point(0, 0, -10), Color(1, 1, 1)),
        ...     shapes=[
        ...         Sphere(),
        ...         Sphere(transform=identity().translate(0, 0, 10)),
        ...     ],
        ... )
        >>> ray = Ray(Point(0, 0, 5), Vector(0, 0, 1))
        >>> intersection = hit(*world.shapes[1].intersect(ray))
        >>> world.shade_hit(intersection)== Color(0.1, 0.1, 0.1)
        True
        """
        return lighting(
            intersection.point,
            self.light,
            intersection.eye_vector,
            intersection.normal_vector,
            intersection.shape.material,
            in_shadow=self.is_shadowed(intersection.over_point),
        )

    def color_at(self, ray):
        # type: (Ray) -> Color
        """
        >>> World().color_at(Ray(Point(0, 0, -5), Vector(0, 1, 0)))
        Color(0, 0, 0)

        >>> actual = World().color_at(Ray(Point(0, 0, -5), Vector(0, 0, 1)))
        >>> expected = Color(0.380665, 0.47583, 0.28550)
        >>> actual == expected
        True

        >>> world = World()
        >>> world.shapes[0].material.ambient = 1
        >>> world.shapes[1].material.ambient = 1
        >>> actual = world.color_at(Ray(Point(0, 0, 0.75), Vector(0, 0, -1)))
        >>> expected = world.shapes[1].material.color
        >>> actual == expected
        True
        """
        intersection = hit(*self.intersect(ray))
        if not intersection:
            return Color(0, 0, 0)
        return self.shade_hit(intersection)

    def is_shadowed(self, point):
        # type: (Matrix) -> bool
        """
        >>> World().is_shadowed(Point(0, 10, 0))
        False

        >>> World().is_shadowed(Point(10, -10, 10))
        True

        >>> World().is_shadowed(Point(-20, 20, -20))
        False

        >>> World().is_shadowed(Point(-2, 2, -2))
        False

        >>> World(
        ...     light=PointLight(Point(0, 0.25, 0), Color(1, 1, 1))
        ... ).is_shadowed(Point(0, 0, 0.5))
        False
        """
        light_vector = self.light.position - point
        distance = light_vector.magnitude
        intersection = hit(*self.intersect(Ray(point, light_vector.normalize())))
        return bool(intersection) and 0 < intersection.t < distance


class Camera:

    def __init__(self, width, height, fov, position=None, target=None, up=None): # pylint: disable = invalid-name
        # type: (int, int, float, Matrix, Matrix, Matrix) -> None
        """
        >>> round(Camera(200, 125, PI / 2).pixel_size, 2)
        0.01

        >>> round(Camera(125, 200, PI / 2).pixel_size, 2)
        0.01
        """
        self.width = width
        self.height = height
        self.fov = fov
        half_view = tan(self.fov / 2)
        aspect = self.width / self.height
        if aspect >= 1:
            self.half_width = half_view
            self.half_height = half_view / aspect
        else:
            self.half_width = half_view * aspect
            self.half_height = half_view
        self.pixel_size = (self.half_width * 2) / self.width
        self.transform = Camera._view_transform(position, target, up)

    def pixel_ray(self, r, c):
        # type: (int, int) -> Ray
        """
        >>> actual = Camera(201, 101, PI / 2).pixel_ray(50, 100)
        >>> expected = Ray(Point(0, 0, 0), Vector(0, 0, -1))
        >>> actual == expected
        True

        >>> actual = Camera(201, 101, PI / 2).pixel_ray(0, 0)
        >>> expected = Ray(Point(0, 0, 0), Vector(-0.66519, 0.33259, -0.66851))
        >>> actual == expected
        True

        >>> actual = Camera(201, 101, PI / 2, identity().translate(0, -2, 5).rotate_y(PI / 4)).pixel_ray(50, 100)
        >>> expected = Ray(Point(0, 2, -5), Vector(sqrt(2)/2, 0, -sqrt(2)/2))
        >>> actual == expected
        True
        """
        # computer the coordinates of the pixel, offsetting to the center of the pixel
        # assumes the camera is at the origin looking at -z, and the canvas is at z=-1
        pixel = Point(
            -self.half_width + (c + 0.5) * self.pixel_size,
            self.half_height - (r + 0.5) * self.pixel_size,
            -1,
        )
        # computer the ray from the camera to the pixel
        pixel = self.transform.inverse() @ pixel
        origin = self.transform.inverse() @ Point(0, 0, 0)
        return Ray(origin, (pixel - origin).normalize())

    def render(self, world):
        # type: (World) -> Canvas
        """
        >>> canvas = Camera(11, 11, PI / 2, Point(0, 0, -5), Point(0, 0, 0), Vector(0, 1, 0)).render(World())
        >>> canvas.get_pixel(5, 5) == Color(0.38066, 0.47583, 0.28550)
        True
        """
        result = Canvas(self.width, self.height)
        for r in range(self.height):
            for c in range(self.width):
                result.set_pixel(r, c, world.color_at(self.pixel_ray(r, c)))
        return result

    @staticmethod
    def _view_transform(position=None, target=None, up=None): # pylint: disable = invalid-name
        # type: (Matrix, Matrix, Matrix) -> Matrix
        """
        >>> actual = Camera._view_transform(Point(0, 0, 0), Point(0, 0, -1), Vector(0, 1, 0))
        >>> expected = identity()
        >>> actual == expected
        True

        >>> actual = Camera._view_transform(Point(0, 0, 0), Point(0, 0, 1), Vector(0, 1, 0))
        >>> expected = identity().scale(-1, 1, -1)
        >>> actual == expected
        True

        >>> actual = Camera._view_transform(Point(0, 0, 8), Point(0, 0, 0), Vector(0, 1, 0))
        >>> expected = identity().translate(0, 0, -8)
        >>> actual == expected
        True

        >>> actual = Camera._view_transform(Point(1, 3, 2), Point(4, -2, 8), Vector(1, 1, 0))
        >>> expected = Matrix([
        ...    [-0.50709, 0.50709, 0.67612, -2.36643],
        ...    [0.76772, 0.60609, 0.12122, -2.82843],
        ...    [-0.35857, 0.59761, -0.71714, 0.0],
        ...    [0, 0, 0, 1]
        ... ])
        >>> actual == expected
        True
        """
        if position is None:
            position = Point(0, 0, 0)
        if target is None:
            target = Point(0, 0, -100)
        if up is None:
            up = Vector(0, 1, 0)
        forward = (target - position).normalize()
        left = forward.cross(up.normalize())
        up = left.cross(forward)
        orientation = Matrix([
            [left.x, left.y, left.z, 0],
            [up.x, up.y, up.z, 0],
            [-forward.x, -forward.y, -forward.z, 0],
            [0, 0, 0, 1],
        ])
        return orientation @ identity().translate(-position.x, -position.y, -position.z)


def demo_vectors():
    # type: () -> None
    position = Point(0, 1, 0)
    velocity = Vector(1, 1.8, 0).normalize() * 11.25
    gravity = Vector(0, -0.1, 0)
    wind = Vector(-0.01, 0, 0)
    canvas = Canvas(900, 550)
    while position.y > 0:
        canvas.set_pixel(canvas.height - position.y, position.x, Color(1, 0, 0))
        position += velocity
        velocity += gravity + wind
    canvas.save(Path('demo_vectors.ppm'))


def demo_transforms():
    # type: () -> None
    position = Point(0, 50, 0)
    rotate = identity().rotate_z(PI / 6)
    canvas = Canvas(150, 150)
    screen_transform = (
        identity()
        .scale(1, -1, 0)
        .translate(canvas.width // 2, canvas.height // 2 - 1, 0)
    )
    for _ in range(12):
        screen_coord = screen_transform @ position
        canvas.set_pixel(screen_coord.y, screen_coord.x, Color(1, 0, 0))
        position = rotate @ position
    canvas.save(Path('demo-transforms.ppm'))


def demo_intersection():
    # type: () -> None
    sphere = Sphere(identity().scale(15, 15, 15))
    canvas = Canvas(60, 60)
    ray_origin = Point(0, 0, 70)
    screen_transform = (
        identity()
        .scale(1, -1, 0)
        .translate(canvas.width // 2, canvas.height // 2 - 1, 0)
    )
    for x in range(canvas.width):
        world_x = x - canvas.width // 2
        for y in range(canvas.height):
            world_y = y - canvas.height // 2
            ray = Ray(ray_origin, (Point(world_x + 0.5, world_y + 0.5, -30) - ray_origin).normalize())
            screen_coord = screen_transform @ Point(world_x, world_y, -30)
            if sphere.intersect(ray):
                canvas.set_pixel(screen_coord.y, screen_coord.x, Color(0, 0, 0))
            else:
                canvas.set_pixel(screen_coord.y, screen_coord.x, Color(1, 1, 1))
    canvas.save(Path('demo-intersection.ppm'))


def demo_lighting():
    # type: () -> None
    material = Material(color=Color(.74, .13, .13))
    sphere = Sphere(identity().scale(20, 20, 20), material=material)
    light = PointLight(Point(70, 70, 70), Color(1, 1, 1))
    eye = Point(0, 0, 60)
    canvas = Canvas(100, 100)
    screen_transform = (
        identity()
        .scale(1, -1, 0)
        .translate(canvas.width // 2, canvas.height // 2 - 1, 0)
    )
    for x in range(canvas.width):
        world_x = x - canvas.width // 2
        for y in range(canvas.height):
            world_y = y - canvas.height // 2
            ray = Ray(eye, Point(world_x, world_y, -30) - eye)
            hit_point = hit(*sphere.intersect(ray))
            screen_coord = screen_transform @ Point(world_x, world_y, -30)
            if not hit_point:
                canvas.set_pixel(screen_coord.y, screen_coord.x, Color(0, 0, 0))
                continue
            point = ray.position(hit_point.t)
            color = lighting(
                point=point,
                light=light,
                eye_vector=-ray.direction,
                normal_vector=hit_point.shape.normal(point),
                material=material,
            )
            canvas.set_pixel(screen_coord.y, screen_coord.x, color)
    canvas.save(Path('demo-lighting.ppm'))


def demo_camera():
    # type: () -> None
    floor = Sphere(
        transform=identity().scale(10, 0.01, 10),
        material=Material(color=Color(1, 0.9, 0.9), specular=0),
    )
    left_wall = Sphere(
        transform=identity().scale(10, 0.01, 10).rotate_x(PI / 2).rotate_y(-PI / 4).translate(0, 0, 5),
        material=floor.material,
    )
    right_wall = Sphere(
        transform=identity().scale(10, 0.01, 10).rotate_x(PI / 2).rotate_y(PI / 4).translate(0, 0, 5),
        material=floor.material,
    )
    middle_sphere = Sphere(
        transform=identity().translate(-0.5, 1, 0.5),
        material=Material(color=Color(0.1, 1, 0.5), diffuse=0.7, specular=0.3),
    )
    right_sphere = Sphere(
        transform=identity().scale(0.5, 0.5, 0.5).translate(1.5, 0.5, -0.5),
        material=Material(color=Color(0.5, 1, 0.1), diffuse=0.7, specular=0.3),
    )
    left_sphere = Sphere(
       transform=identity().scale(0.33, 0.33, 0.33).translate(-1.5, 0.33, -0.75),
       material=Material(color=Color(1, 0.8, 0.1), diffuse=0.7, specular=0.3),
    )
    world = World(
        light=PointLight(Point(-10, 10, -10), Color(1, 1, 1)),
        shapes=[
            floor, left_wall, right_wall,
            middle_sphere, left_sphere, right_sphere,
        ],
    )
    camera = Camera(
        100, 50, PI / 3,
        Point(0, 1.5, -5),
        Point(0, 1, 0),
        Point(0, 1, 0),
    )
    camera.render(world).save(Path('demo-camera.ppm'))


def demo_axes(width, height):
    # type: (int, int) -> None
    world = World(
        light=PointLight(Point(10, 10, 10), Color(1, 1, 1)),
        shapes=[
            Sphere( # xy "plane"
                material=Material(color=Color(0.5, 0.5, 1), specular=0),
                transform=identity().scale(50, 50, 50).translate(0, 0, -53),
            ),
            Sphere( # yz "plane"
                material=Material(color=Color(1, 0.5, 0.5), specular=0),
                transform=identity().scale(50, 50, 50).translate(-53, 0, 0),
            ),
            Sphere( # xz "plane"
                material=Material(color=Color(0.5, 1, 0.5), specular=0),
                transform=identity().scale(50, 50, 50).translate(0, -53, 0),
            ),
            Sphere( # x sphere (red)
                material=Material(color=Color(1, 0, 0)),
                transform=identity().translate(2, 0, 0),
            ),
            Sphere( # y sphere (green)
                material=Material(color=Color(0, 1, 0)),
                transform=identity().translate(0, 2, 0),
            ),
            Sphere( # z sphere (blue)
                material=Material(color=Color(0, 0, 1)),
                transform=identity().translate(0, 0, 2),
            ),
        ],
    )
    camera = Camera(
        width, height, PI / 3,
        Point(0, 3, 10),
        Point(0, 0, 0),
        Point(0, 1, 0),
    )
    camera.render(world).save(Path('demo-axes.ppm'))


def main():
    # type: () -> None
    demo_axes(100, 50)


if __name__ == '__main__':
    main()
