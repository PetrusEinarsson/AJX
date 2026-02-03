import jax
import jax.numpy as jnp
from ajx.block_sparse.base import BlockMatrixBase


from typing import List, Dict, Tuple
from itertools import combinations_with_replacement, product, accumulate
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class VBRMatrix(BlockMatrixBase):
    """A variable block row matrix format."""

    def __init__(
        self,
        data: jax.Array,
        col_indices: Tuple[int],
        row_ptr: Tuple[int],
        row_sizes: Tuple[int],
        col_sizes: Tuple[int],
    ):
        self.data = data
        self.col_indices = col_indices
        self.row_ptr = row_ptr
        self.row_sizes = row_sizes
        self.col_sizes = col_sizes

        self.row_begin_indices = list(accumulate([0, *row_sizes]))
        self.col_begin_indices = list(accumulate([0, *col_sizes]))

        pass

    @property
    def n_rows(self):
        return len(self.row_sizes)

    @property
    def n_cols(self):
        return len(self.col_sizes)

    @property
    def shape(self):
        rows = sum(r for r in self.row_sizes)
        cols = sum(c for c in self.col_sizes)
        return (rows, cols)

    def tree_flatten(self):
        children = (self.data,)
        aux_data = (self.col_indices, self.row_ptr, self.row_sizes, self.col_sizes)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    def to_scalar_matrix(self):
        full_mat = jnp.zeros(self.shape)
        slice_begin = 0
        col_id = 0
        for k in range(len(self.row_ptr) - 1):
            row_id = k
            for i in range(self.row_ptr[k], self.row_ptr[k + 1]):
                col_id = self.col_indices[i]
                shape = (self.row_sizes[row_id], self.col_sizes[col_id])
                slice_end = slice_begin + shape[0] * shape[1]
                mat = self.data[slice_begin:slice_end].reshape(shape)
                row_ptr = self.row_begin_indices[row_id]
                col_ptr = self.col_begin_indices[col_id]
                full_mat = full_mat.at[
                    row_ptr : row_ptr + shape[0], col_ptr : col_ptr + shape[1]
                ].set(mat)
                slice_begin = slice_end

        return full_mat

    def mul_vector(self, vec):
        res = jnp.zeros(sum(self.row_sizes))
        slice_begin = 0
        for k in range(len(self.row_ptr) - 1):
            row_id = k
            for i in range(self.row_ptr[k], self.row_ptr[k + 1]):
                col_id = self.col_indices[i]
                shape = (self.row_sizes[row_id], self.col_sizes[col_id])
                slice_end = slice_begin + shape[0] * shape[1]
                mat = self.data[slice_begin:slice_end].reshape(shape)

                col_slice_begin = self.col_begin_indices[col_id]
                col_slice_end = self.col_begin_indices[col_id + 1]
                row_slice_begin = self.row_begin_indices[row_id]
                row_slice_end = self.row_begin_indices[row_id + 1]

                vec_data = vec[col_slice_begin:col_slice_end]

                local_result = mat @ vec_data
                res = res.at[row_slice_begin:row_slice_end].set(
                    res[row_slice_begin:row_slice_end] + local_result
                )
                slice_begin = slice_end
        return res

    def vector_mul(self, vec):
        res = jnp.zeros(sum(self.col_sizes))
        slice_begin = 0
        for k in range(len(self.row_ptr) - 1):
            row_id = k
            for i in range(self.row_ptr[k], self.row_ptr[k + 1]):
                col_id = self.col_indices[i]
                shape = (self.row_sizes[row_id], self.col_sizes[col_id])
                slice_end = slice_begin + shape[0] * shape[1]
                mat = self.data[slice_begin:slice_end].reshape(shape)

                col_slice_begin = self.col_begin_indices[col_id]
                col_slice_end = self.col_begin_indices[col_id + 1]
                row_slice_begin = self.row_begin_indices[row_id]
                row_slice_end = self.row_begin_indices[row_id + 1]

                vec_data = vec[row_slice_begin:row_slice_end]

                local_result = vec_data @ mat
                res = res.at[col_slice_begin:col_slice_end].set(
                    res[col_slice_begin:col_slice_end] + local_result
                )
                slice_begin = slice_end

        return res

    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure()
        scalar_repr = self.to_scalar_matrix()
        ignore_imshow = False
        while isinstance(scalar_repr, jax.core.Tracer):
            if hasattr(scalar_repr, "val"):
                scalar_repr = scalar_repr.val[0]
            else:
                ignore_imshow = True
                break
        if not ignore_imshow:
            plt.imshow(
                jnp.abs(self.to_scalar_matrix()) < 1e-22,
                cmap="YlGnBu",
                aspect="auto",
                interpolation="none",
            )
            plt.imshow(
                jnp.abs(jnp.abs(self.to_scalar_matrix()) - 1) < 1e-6,
                cmap="Reds",
                aspect="auto",
                interpolation="none",
                alpha=0.6,
            )
        else:
            plt.gca().invert_yaxis()
        plt.tight_layout()

        for i, (row_begin_prev, row_begin) in enumerate(
            zip(self.row_begin_indices, self.row_begin_indices[1:])
        ):
            plt.axhline(row_begin - 0.5, linewidth=2)
            for j in range(self.row_sizes[i]):
                pass
            plt.text(
                0,
                (row_begin_prev + row_begin) * 0.5 - 0.5,
                i,
                verticalalignment="center",
            )

        for i, (col_begin_prev, col_begin) in enumerate(
            zip(self.col_begin_indices, self.col_begin_indices[1:])
        ):
            plt.axvline(col_begin - 0.5)
            plt.text(
                (col_begin_prev + col_begin) * 0.5 - 0.5,
                0,
                i,
                horizontalalignment="center",
            )

        plt.show()


if __name__ == "__main__":
    jnp.set_printoptions(edgeitems=30, linewidth=1000)
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(4)
    key, key1 = jax.random.split(key)
    key, key2 = jax.random.split(key)

    A = jnp.ones([2, 2]) * 1.0
    B = jnp.ones([2, 2]) * 2.0
    C = jnp.ones([4, 1]) * 3.0
    D = jnp.ones([2, 3]) * 4.0

    data = jnp.concatenate([A, B, C, D], axis=None)
    col_indices = (0, 1, 2, 1)
    row_ptr = (0, 2, 3, 4)
    row_sizes = (2, 2, 4)
    col_sizes = (2, 1, 3)
    bcsr_mat = VBRMatrix(data, col_indices, row_ptr, row_sizes, col_sizes)

    vec = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    vec2 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    res = bcsr_mat.mul_vector(vec)
    res2 = bcsr_mat.vector_mul(vec2)
    print(bcsr_mat.to_scalar_matrix())
