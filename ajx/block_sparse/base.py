import abc
import jax


class BlockMatrixBase(abc.ABC):
    """
    Base class for different block matrices. Here, a block matrix has
    shape (norows, nocols, nirows, nicols).
    norows - number of outer rows
    nocols - number of outer columns
    nirows - number of inner rows
    nicols - number of inner columns
    """

    # Not abstract methods because not all sparse classes implement them
    def __neg__(self):
        raise NotImplementedError(f"{self.__class__}.__neg__")

    def __pos__(self):
        raise NotImplementedError(f"{self.__class__}.__pos__")

    def __matmul__(self, other):
        raise NotImplementedError(f"{self.__class__}.__matmul__")

    def __rmatmul__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rmatmul__")

    def __mul__(self, other):
        raise NotImplementedError(f"{self.__class__}.__mul__")

    def __rmul__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rmul__")

    def __add__(self, other):
        raise NotImplementedError(f"{self.__class__}.__add__")

    def __radd__(self, other):
        raise NotImplementedError(f"{self.__class__}.__radd__")

    def __sub__(self, other):
        raise NotImplementedError(f"{self.__class__}.__sub__")

    def __rsub__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rsub__")

    def __getitem__(self, item):
        raise NotImplementedError(f"{self.__class__}.__getitem__")
