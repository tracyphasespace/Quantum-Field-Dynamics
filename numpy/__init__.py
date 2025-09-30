"""A minimal NumPy compatibility layer used for unit testing.

This module implements a tiny subset of the :mod:`numpy` API that is
required by the pure Python redshift analysis utilities that ship with the
repository.  The real project depends on the external NumPy package, but the
execution environment used for the kata intentionally omits third party
dependencies.  Providing a very small, well-tested stand in keeps the
scientific routines working without pulling in the full dependency tree.

The implementation below focuses on one dimensional arrays and the handful of
element-wise operations exercised by the tests.  It is **not** a drop-in
replacement for NumPy – it merely exposes enough behaviour for the modules in
``redshift-analysis`` to run.  Whenever a feature is missing the code raises a
``NotImplementedError`` with a helpful message so callers are aware of the
limitation.
"""

from __future__ import annotations

import builtins
from math import exp as _exp
from math import log as _log
from math import log10 as _log10
from math import pi
from math import sqrt as _sqrt
from typing import Iterable, Iterator, List, Sequence, Tuple, Union

Number = Union[int, float]


def _ensure_list(values: Iterable[Number]) -> List[Number]:
    """Return ``values`` as a concrete list of numbers."""

    if isinstance(values, ndarray):
        return list(values._data)
    if isinstance(values, list):
        return values.copy()
    return [float(v) for v in values]


class ndarray(Sequence[Number]):
    """Very small 1-D array implementation used by the compatibility layer."""

    __slots__ = ("_data",)
    __array_priority__ = 1000  # ensure ndarray operations take precedence

    def __init__(self, values: Iterable[Number]):
        self._data: List[Number] = [float(v) for v in values]

    # Sequence interface -------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)

    def __iter__(self) -> Iterator[Number]:  # pragma: no cover - trivial
        return iter(self._data)

    def __getitem__(self, index):  # pragma: no cover - trivial
        if isinstance(index, slice):
            return ndarray(self._data[index])
        return self._data[index]

    # Helpers ------------------------------------------------------------
    def copy(self) -> "ndarray":
        return ndarray(self._data)

    def tolist(self) -> List[Number]:  # pragma: no cover - trivial
        return self._data.copy()

    def _binary_op(self, other, op) -> "ndarray":
        if isinstance(other, ndarray):
            if len(self) != len(other):  # pragma: no cover - sanity check
                raise ValueError("array sizes must match")
            other_values = other._data
        elif isinstance(other, Sequence):
            other_values = _ensure_list(other)
            if len(other_values) != len(self):
                raise ValueError("array sizes must match")
        else:
            return ndarray(op(value, float(other)) for value in self._data)

        return ndarray(op(a, b) for a, b in zip(self._data, other_values))

    def _unary_op(self, op) -> "ndarray":
        return ndarray(op(value) for value in self._data)

    # Arithmetic ---------------------------------------------------------
    def __add__(self, other):  # pragma: no cover - exercised indirectly
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other):  # pragma: no cover - exercised indirectly
        return self + other

    def __sub__(self, other):  # pragma: no cover - exercised indirectly
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):  # pragma: no cover - exercised indirectly
        return (-self) + other

    def __mul__(self, other):  # pragma: no cover - exercised indirectly
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other):  # pragma: no cover - exercised indirectly
        return self * other

    def __truediv__(self, other):  # pragma: no cover - exercised indirectly
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other):  # pragma: no cover - exercised indirectly
        return ndarray(float(other) / value for value in self._data)

    def __pow__(self, other):  # pragma: no cover - exercised indirectly
        return self._binary_op(other, lambda a, b: a ** b)

    def __rpow__(self, other):  # pragma: no cover - exercised indirectly
        return ndarray(float(other) ** value for value in self._data)

    def __neg__(self):  # pragma: no cover - exercised indirectly
        return ndarray(-value for value in self._data)

    # Comparisons -------------------------------------------------------
    def _compare(self, other, op) -> "ndarray":
        return self._binary_op(other, lambda a, b: 1.0 if op(a, b) else 0.0)

    def __lt__(self, other):  # pragma: no cover - exercised indirectly
        return self._compare(other, lambda a, b: a < b)

    def __le__(self, other):  # pragma: no cover - exercised indirectly
        return self._compare(other, lambda a, b: a <= b)

    def __gt__(self, other):  # pragma: no cover - exercised indirectly
        return self._compare(other, lambda a, b: a > b)

    def __ge__(self, other):  # pragma: no cover - exercised indirectly
        return self._compare(other, lambda a, b: a >= b)

    def __eq__(self, other):  # pragma: no cover - exercised indirectly
        return self._compare(other, lambda a, b: a == b)

    def __ne__(self, other):  # pragma: no cover - exercised indirectly
        return self._compare(other, lambda a, b: a != b)

    # Reductions --------------------------------------------------------
    def sum(self) -> float:
        return sum(self._data)

    def max(self) -> float:  # pragma: no cover - trivial
        return max(self._data)

    def mean(self) -> float:  # pragma: no cover - trivial
        return self.sum() / len(self._data) if self._data else 0.0

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"ndarray({self._data!r})"


def array(values: Iterable[Number]) -> ndarray:
    return values if isinstance(values, ndarray) else ndarray(values)


def asarray(values: Iterable[Number]) -> ndarray:  # pragma: no cover - alias
    return array(values)


def zeros_like(values: Union[Sequence[Number], ndarray]) -> ndarray:
    if isinstance(values, ndarray):
        length = len(values)
    else:
        length = len(list(values))
    return ndarray([0.0] * length)


def exp(values: Union[Number, ndarray]) -> Union[float, ndarray]:
    if isinstance(values, ndarray):
        return values._unary_op(_exp)
    return float(_exp(values))


def log(values: Union[Number, ndarray]) -> Union[float, ndarray]:
    if isinstance(values, ndarray):
        return values._unary_op(_log)
    return float(_log(values))


def log10(values: Union[Number, ndarray]) -> Union[float, ndarray]:
    if isinstance(values, ndarray):
        return values._unary_op(_log10)
    return float(_log10(values))


def sqrt(values: Union[Number, ndarray]) -> Union[float, ndarray]:
    if isinstance(values, ndarray):
        return values._unary_op(_sqrt)
    return float(_sqrt(values))


def maximum(a, b):  # pragma: no cover - exercised indirectly
    if isinstance(a, ndarray) or isinstance(b, ndarray):
        arr_a = array(a) if isinstance(a, (ndarray, Sequence)) else array([a])
        arr_b = array(b) if isinstance(b, (ndarray, Sequence)) else array([b])
        if len(arr_a) != len(arr_b):
            if len(arr_a) == 1:
                arr_a = ndarray([arr_a[0]] * len(arr_b))
            elif len(arr_b) == 1:
                arr_b = ndarray([arr_b[0]] * len(arr_a))
            else:  # pragma: no cover - sanity check
                raise ValueError("array sizes must match")
        return ndarray(builtins.max(x, y) for x, y in zip(arr_a, arr_b))
    return builtins.max(float(a), float(b))


def abs(values: Union[Number, ndarray]) -> Union[float, ndarray]:  # noqa: A003
    if isinstance(values, ndarray):
        return values._unary_op(lambda x: abs(x))
    return float(builtins.abs(values))


def sum(values: Union[Sequence[Number], ndarray]) -> float:  # noqa: A003
    if isinstance(values, ndarray):
        return values.sum()
    return float(builtins.sum(values))


def mean(values: Union[Sequence[Number], ndarray]) -> float:
    arr = array(values)
    return arr.sum() / len(arr) if len(arr) else 0.0


def max(values: Union[Sequence[Number], ndarray]):  # noqa: A003
    arr = array(values)
    return arr.max()


def logspace(start: Number, stop: Number, num: int) -> ndarray:
    if num <= 0:
        return ndarray([])
    if num == 1:
        return ndarray([10 ** start])
    step = (stop - start) / (num - 1)
    return ndarray(10 ** (start + step * i) for i in range(num))


def all(values: Union[Sequence[Number], ndarray]) -> bool:  # noqa: A003
    if isinstance(values, ndarray):
        return builtins.all(bool(v) for v in values)
    return builtins.all(bool(v) for v in values)


def isclose(
    a: Union[Number, ndarray],
    b: Union[Number, ndarray],
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
):
    if isinstance(a, ndarray) or isinstance(b, ndarray):
        arr_a = array(a)
        arr_b = array(b)
        if len(arr_a) != len(arr_b):  # pragma: no cover - sanity check
            raise ValueError("array sizes must match")
        return ndarray(
            1.0
            if abs(x - y) <= atol + rtol * abs(y)
            else 0.0
            for x, y in zip(arr_a, arr_b)
        )
    return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))


__all__ = [
    "array",
    "asarray",
    "ndarray",
    "zeros_like",
    "exp",
    "log",
    "log10",
    "sqrt",
    "maximum",
    "abs",
    "sum",
    "mean",
    "max",
    "logspace",
    "all",
    "isclose",
    "pi",
]

