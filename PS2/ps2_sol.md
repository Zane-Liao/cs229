# PS1 Solution

1.

(a)

- Data Sets A Converge quickly, Data Sets B No convergence.

(b)

- Dataset A is linearly inseparable and has a unique minimum value,but data set B is linearly separable.
In mathematics, when a linearlyseparable data set appears without adding a regularization term,
the logarithmic loss function of logistic regression has no minimum value,
which will cause non-convergence.
$\mathcal{L}(\theta) = -\sum_{i=1}^{n} \left[y_i \log(p_i) + (1 - y_i) \log(1 - p_i)\right]$

(c)

(i and iii)
- It is not possible to find a unique minimum value.
(ii and v)
- They can only alleviate the problem of divergence and
non-convergence to a certain extent, but they are similar to
a kind of "weak regularization" and are not strictly convergent
in the mathematical sense.
(iv)
- Regularization is strictly a way to make the loss function converge
Why?
Mathematically speaking, our logistic regression loss function is
semi-positive, which means that it is not guaranteed to find a global
unique minimum, but our regularization term
$\|\theta\|_2^2 = \theta^T\theta$
Is positive definite, which means that it is a strictly convex function,
and we can get a strictly convex function plus a convex function
equals a strictly convex function, so adding a regularization term
can guarantee a global unique minimum.
(d)
- It is not easily affected because Hinge-Loss only cares about
the data within the boundary, that is, it only penalizes the data
within the boundary, and no longer penalizes the data outside the boundary.
From a mathematical point of view, $y(w^T x + b) \geq 1$, the loss is 0.
2.

(a)