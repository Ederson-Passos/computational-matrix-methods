# Lecture 04: SVD

## Theorem 1 (SVD)

$$ Ax = \sigma y $$

* Was demonstrated during the lecture 04;
* Two forms to calcute norm-2;
* Notes from proof:

$$ \|x\|_2 = \|y\|_2 = 1 $$
where, $\sigma = \|A\|_2$.

From theorem 2, we know that exists $V_2 \in \mathbb{R}^{m \times (m-1)}$ and $U_2 \in \mathbb{R}^{m \times (m-1)}$, such that $V = [x \mid V_2] \in \mathbb{R}^{m \times n}$ and $U = [y \mid U_2] \in \mathbb{R}^{m \times m}$...

As

$$ \left\| A_1 \begin{bmatrix} \sigma \\ w \end{bmatrix} \right\|_2^2 \ge (\sigma^2 + w^T w)^2 $$

we will have that $ \left\| A_1 \right\|_2^2 \ge (\sigma^2 + w^T w)^2 $. But, by definition, we know that ...

* The theorem will tell that exists two solutions (matrix), but not uniqueness.