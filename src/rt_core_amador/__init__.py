"""Pure-Python mathematical utilities for rt-core-amador."""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable, Iterable, Sequence
from typing import TypeAlias

Number: TypeAlias = int | float
ComplexPoint: TypeAlias = tuple[complex, ...]


def _flatten_rectangular(value: object) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Convert a scalar or rectangular list/tuple into flat data and its shape."""
    if isinstance(value, bool):
        return (float(value),), ()
    if isinstance(value, (int, float)):
        return (float(value),), ()
    if not isinstance(value, (list, tuple)):
        raise TypeError("Tensor data must be a number or a nested list/tuple.")
    if not value:
        return (), (0,)

    children = [_flatten_rectangular(item) for item in value]
    child_shape = children[0][1]
    if any(shape != child_shape for _, shape in children):
        raise ValueError("Tensor data must be rectangular.")
    return tuple(item for flat, _ in children for item in flat), (len(value),) + child_shape


class AmadorTensor:
    """A compact dense tensor with basic shape-checked arithmetic."""

    __slots__ = ("_data", "_shape")

    def __init__(self, data: object) -> None:
        flat, shape = _flatten_rectangular(data)
        self._data = flat
        self._shape = shape

    @classmethod
    def _from_flat(cls, data: Iterable[float], shape: tuple[int, ...]) -> AmadorTensor:
        instance = cls.__new__(cls)
        instance._data = tuple(float(value) for value in data)
        instance._shape = shape
        return instance

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def data(self) -> tuple[float, ...]:
        return self._data

    def to_list(self) -> object:
        if not self._shape:
            return self._data[0]

        def build(offset: int, dimensions: tuple[int, ...]) -> tuple[object, int]:
            if len(dimensions) == 1:
                end = offset + dimensions[0]
                return list(self._data[offset:end]), end
            result = []
            for _ in range(dimensions[0]):
                item, offset = build(offset, dimensions[1:])
                result.append(item)
            return result, offset

        result, _ = build(0, self._shape)
        return result

    def _elementwise(self, other: Number | AmadorTensor, operation: Callable[[float, float], float]) -> AmadorTensor:
        if isinstance(other, AmadorTensor):
            if self.shape != other.shape:
                raise ValueError(f"Tensor shapes must match; got {self.shape} and {other.shape}.")
            values = (operation(left, right) for left, right in zip(self._data, other._data))
        elif isinstance(other, (int, float)) and not isinstance(other, bool):
            scalar = float(other)
            values = (operation(value, scalar) for value in self._data)
        else:
            return NotImplemented
        return self._from_flat(values, self._shape)

    def __add__(self, other: Number | AmadorTensor) -> AmadorTensor:
        return self._elementwise(other, lambda left, right: left + right)

    def __radd__(self, other: Number) -> AmadorTensor:
        return self + other

    def __sub__(self, other: Number | AmadorTensor) -> AmadorTensor:
        return self._elementwise(other, lambda left, right: left - right)

    def __rsub__(self, other: Number) -> AmadorTensor:
        return self._from_flat((other - value for value in self._data), self._shape)

    def __mul__(self, other: Number | AmadorTensor) -> AmadorTensor:
        return self._elementwise(other, lambda left, right: left * right)

    def __rmul__(self, other: Number) -> AmadorTensor:
        return self * other

    def __truediv__(self, other: Number | AmadorTensor) -> AmadorTensor:
        return self._elementwise(other, lambda left, right: left / right)

    def __neg__(self) -> AmadorTensor:
        return self._from_flat((-value for value in self._data), self._shape)

    def dot(self, other: AmadorTensor) -> float:
        if self.ndim != 1 or other.ndim != 1:
            raise ValueError("dot() is defined for one-dimensional tensors.")
        if self.shape != other.shape:
            raise ValueError(f"Vector lengths must match; got {self.shape} and {other.shape}.")
        return sum(left * right for left, right in zip(self._data, other._data))

    def matmul(self, other: AmadorTensor) -> AmadorTensor:
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("matmul() is defined for two-dimensional tensors.")
        rows, inner = self.shape
        other_inner, columns = other.shape
        if inner != other_inner:
            raise ValueError(f"Incompatible matrix shapes {self.shape} and {other.shape}.")
        values = (
            sum(
                self._data[row * inner + index] * other._data[index * columns + column]
                for index in range(inner)
            )
            for row in range(rows)
            for column in range(columns)
        )
        return self._from_flat(values, (rows, columns))

    def __repr__(self) -> str:
        return f"AmadorTensor(data={self.to_list()!r}, shape={self.shape!r})"


class AmadorTransform:
    """Phase-modulated direct discrete Fourier transform.

    Frequency bin k is multiplied by exp(i * phase * k). This convention is
    defined by this package and is not intended as a claim about prior work.
    """

    def __init__(self, phase: float = 0.0) -> None:
        if not math.isfinite(phase):
            raise ValueError("phase must be finite.")
        self.phase = float(phase)

    def forward(self, values: Sequence[Number | complex]) -> tuple[complex, ...]:
        if not values:
            raise ValueError("Transform input cannot be empty.")
        count = len(values)
        return tuple(
            sum(
                complex(value) * cmath.exp(-2j * math.pi * frequency * index / count)
                for index, value in enumerate(values)
            )
            / count
            * cmath.exp(1j * self.phase * frequency)
            for frequency in range(count)
        )

    def inverse(self, coefficients: Sequence[Number | complex]) -> tuple[complex, ...]:
        if not coefficients:
            raise ValueError("Transform input cannot be empty.")
        count = len(coefficients)
        return tuple(
            sum(
                complex(coefficient)
                * cmath.exp(-1j * self.phase * frequency)
                * cmath.exp(2j * math.pi * frequency * index / count)
                for frequency, coefficient in enumerate(coefficients)
            )
            for index in range(count)
        )
class NavierStokesOptimizer:
    """One explicit projection-method step for periodic 2D velocity grids."""

    def __init__(
        self,
        viscosity: float = 0.01,
        dx: float = 1.0,
        dy: float = 1.0,
        pressure_iterations: int = 50,
    ) -> None:
        if not math.isfinite(viscosity) or viscosity < 0:
            raise ValueError("viscosity must be finite and non-negative.")
        if not math.isfinite(dx) or dx <= 0 or not math.isfinite(dy) or dy <= 0:
            raise ValueError("dx and dy must be finite and positive.")
        if (
            isinstance(pressure_iterations, bool)
            or not isinstance(pressure_iterations, int)
            or pressure_iterations < 1
        ):
            raise ValueError("pressure_iterations must be a positive integer.")
        self.viscosity = float(viscosity)
        self.dx = float(dx)
        self.dy = float(dy)
        self.pressure_iterations = int(pressure_iterations)

    @staticmethod
    def _validate_grid(grid: Sequence[Sequence[Number]], name: str) -> list[list[float]]:
        values = [list(map(float, row)) for row in grid]
        if not values or not values[0] or any(len(row) != len(values[0]) for row in values):
            raise ValueError(f"{name} must be a non-empty rectangular 2D grid.")
        if any(not math.isfinite(value) for row in values for value in row):
            raise ValueError(f"{name} must contain only finite values.")
        return values

    def step(
        self,
        u: Sequence[Sequence[Number]],
        v: Sequence[Sequence[Number]],
        dt: float,
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        """Advance velocity by dt and return new u, v, and pressure grids."""
        u_grid = self._validate_grid(u, "u")
        v_grid = self._validate_grid(v, "v")
        rows, columns = len(u_grid), len(u_grid[0])
        if len(v_grid) != rows or len(v_grid[0]) != columns:
            raise ValueError("u and v must have identical grid dimensions.")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")

        def at(grid: list[list[float]], row: int, column: int) -> float:
            return grid[row % rows][column % columns]

        dx2, dy2 = self.dx * self.dx, self.dy * self.dy
        u_star = [[0.0] * columns for _ in range(rows)]
        v_star = [[0.0] * columns for _ in range(rows)]
        divergence = [[0.0] * columns for _ in range(rows)]
        for row in range(rows):
            for column in range(columns):
                uc, vc = u_grid[row][column], v_grid[row][column]
                du_dx = (at(u_grid, row, column + 1) - at(u_grid, row, column - 1)) / (2 * self.dx)
                du_dy = (at(u_grid, row + 1, column) - at(u_grid, row - 1, column)) / (2 * self.dy)
                dv_dx = (at(v_grid, row, column + 1) - at(v_grid, row, column - 1)) / (2 * self.dx)
                dv_dy = (at(v_grid, row + 1, column) - at(v_grid, row - 1, column)) / (2 * self.dy)
                lap_u = (
                    (at(u_grid, row, column + 1) - 2 * uc + at(u_grid, row, column - 1)) / dx2
                    + (at(u_grid, row + 1, column) - 2 * uc + at(u_grid, row - 1, column)) / dy2
                )
                lap_v = (
                    (at(v_grid, row, column + 1) - 2 * vc + at(v_grid, row, column - 1)) / dx2
                    + (at(v_grid, row + 1, column) - 2 * vc + at(v_grid, row - 1, column)) / dy2
                )
                u_star[row][column] = uc + dt * (-uc * du_dx - vc * du_dy + self.viscosity * lap_u)
                v_star[row][column] = vc + dt * (-uc * dv_dx - vc * dv_dy + self.viscosity * lap_v)
                divergence[row][column] = (
                    (at(u_star, row, column + 1) - at(u_star, row, column - 1)) / (2 * self.dx)
                    + (at(v_star, row + 1, column) - at(v_star, row - 1, column)) / (2 * self.dy)
                )

        pressure = [[0.0] * columns for _ in range(rows)]
        pressure_next = [[0.0] * columns for _ in range(rows)]
        denominator = 2 / dx2 + 2 / dy2
        for _ in range(self.pressure_iterations):
            for row in range(rows):
                for column in range(columns):
                    pressure_next[row][column] = (
                        (at(pressure, row, column + 1) + at(pressure, row, column - 1)) / dx2
                        + (at(pressure, row + 1, column) + at(pressure, row - 1, column)) / dy2
                        - divergence[row][column] / dt
                    ) / denominator
            pressure, pressure_next = pressure_next, pressure

        u_next = [[0.0] * columns for _ in range(rows)]
        v_next = [[0.0] * columns for _ in range(rows)]
        for row in range(rows):
            for column in range(columns):
                grad_p_x = (at(pressure, row, column + 1) - at(pressure, row, column - 1)) / (2 * self.dx)
                grad_p_y = (at(pressure, row + 1, column) - at(pressure, row - 1, column)) / (2 * self.dy)
                u_next[row][column] = u_star[row][column] - dt * grad_p_x
                v_next[row][column] = v_star[row][column] - dt * grad_p_y
        return u_next, v_next, pressure


class KahlerManifold:
    """Finite-difference Hermitian metric generated by a real potential."""

    def __init__(
        self,
        complex_dimension: int,
        potential: Callable[[ComplexPoint], float] | None = None,
        step_size: float = 1e-4,
    ) -> None:
        if (
            isinstance(complex_dimension, bool)
            or not isinstance(complex_dimension, int)
            or complex_dimension < 1
        ):
            raise ValueError("complex_dimension must be a positive integer.")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be finite and positive.")
        self.complex_dimension = int(complex_dimension)
        self.potential = (
            potential
            if potential is not None
            else lambda point: sum(abs(value) ** 2 for value in point)
        )
        self.step_size = float(step_size)

    def _evaluate(self, point: list[complex]) -> float:
        result = float(self.potential(tuple(point)))
        if not math.isfinite(result):
            raise ValueError("The potential must return a finite real number.")
        return result

    def metric_at(self, point: Sequence[complex]) -> tuple[tuple[complex, ...], ...]:
        """Estimate d²φ/dz_i d(conjugate(z_j)) at a point."""
        if len(point) != self.complex_dimension:
            raise ValueError(f"Expected a point of dimension {self.complex_dimension}.")
        base = [complex(value) for value in point]
        h = self.step_size
        metric: list[tuple[complex, ...]] = []

        def shifted(index: int, delta: complex) -> list[complex]:
            result = base.copy()
            result[index] += delta
            return result

        def second_derivative(
            first_index: int,
            first_delta: complex,
            second_index: int,
            second_delta: complex,
        ) -> float:
            if first_index == second_index and first_delta == second_delta:
                plus = shifted(first_index, first_delta)
                minus = shifted(first_index, -first_delta)
                return (self._evaluate(plus) - 2 * self._evaluate(base) + self._evaluate(minus)) / (h * h)
            pp = base.copy()
            pm = base.copy()
            mp = base.copy()
            mm = base.copy()
            pp[first_index] += first_delta
            pp[second_index] += second_delta
            pm[first_index] += first_delta
            pm[second_index] -= second_delta
            mp[first_index] -= first_delta
            mp[second_index] += second_delta
            mm[first_index] -= first_delta
            mm[second_index] -= second_delta
            return (
                self._evaluate(pp) - self._evaluate(pm) - self._evaluate(mp) + self._evaluate(mm)
            ) / (4 * h * h)

        for i in range(self.complex_dimension):
            row: list[complex] = []
            for j in range(self.complex_dimension):
                dxx = second_derivative(i, h, j, h)
                dyy = second_derivative(i, 1j * h, j, 1j * h)
                dxy = second_derivative(i, h, j, 1j * h)
                dyx = second_derivative(i, 1j * h, j, h)
                real_part = (dxx + dyy) / 4
                imag_part = (dxy - dyx) / 4
                row.append(complex(real_part, imag_part))
            metric.append(tuple(row))
        return tuple(metric)

    def norm_squared(self, point: Sequence[complex], tangent: Sequence[complex]) -> float:
        if len(tangent) != self.complex_dimension:
            raise ValueError(f"Expected a tangent vector of dimension {self.complex_dimension}.")
        metric = self.metric_at(point)
        value = sum(
            metric[i][j] * complex(tangent[i]) * complex(tangent[j]).conjugate()
            for i in range(self.complex_dimension)
            for j in range(self.complex_dimension)
        )
        if abs(value.imag) > 1e-7:
            raise ValueError("Computed metric norm is not real; check that the potential is valid.")
        return float(value.real)


__all__ = [
    "AmadorTensor",
    "AmadorTransform",
    "KahlerManifold",
    "NavierStokesOptimizer",
]
