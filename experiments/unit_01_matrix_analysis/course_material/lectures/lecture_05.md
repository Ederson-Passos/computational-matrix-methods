## Lecture 05: SVD (complete & reduced)

* Complete SVD:

* Reconstructed Matrix $ A = U \Sigma V^T$:

```

A = np.array([
    [1, 1],
    [0, 0],
    [2, 2]
])

r = np.linalg.matrix_rank(A)

```

* SVD Decomposition:

```

U, S, Vt = np.linalg.svd(A)
# where S generate the singular values, U generate the line
# space, and V generate the column space.
# Asemble the sigma matrix, such diangonal are non null.

```

* We will get only the great k values of sigma.
* We obtained the $ A_{k} $ matrix, such that is the best aproximation of A in terms of rank k.
* So, we can obtain the $ U_{k} $ matrix, that is the best aproximation of U in terms of rank k.
* And the $ V_{k} $ matrix, that is the best aproximation of V in terms of rank k.

$$ A_{k} = U_{k} \Sigma_{k} V_{k}^T $$

* The norm-F $ \left\| A - A_{k} \right\|_F $ is mandatory.

## Orthogonal Projection and Distance between subspaces

* The orthogonal projection of a vector $ y $ onto a subspace $ \mathcal{V} $ spanned by an orthonormal basis $ \{v_1, v_2, ..., v_k\} $ is given by:

$$ P_{V} y = \sum_{i=1}^{k} \langle y, v_i \rangle v_i $$

where $ \langle y, v_i \rangle = y^T v_i $ is the inner product (dot product) of vectors $ y $ and $ v_i $.
