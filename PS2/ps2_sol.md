# PS2 Solution

### 1.

##### (a)

- Data Sets A Converge quickly, Data Sets B No convergence.

##### (b)

- Dataset A is linearly inseparable and has a unique minimum value,but data set B is linearly separable.
In mathematics, when a linearlyseparable data set appears without adding a regularization term,
the logarithmic loss function of logistic regression has no minimum value,
which will cause non-convergence.
$\mathcal{L}(\theta) = -\sum_{i=1}^{n} \left[y_i \log(p_i) + (1 - y_i) \log(1 - p_i)\right]$

##### (c)

1. (i and iii)
- It is not possible to find a unique minimum value.
2. (ii and v)
- They can only alleviate the problem of divergence and
non-convergence to a certain extent, but they are similar to
a kind of "weak regularization" and are not strictly convergent
in the mathematical sense.
3. (iv)
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
##### (d)
- It is not easily affected because Hinge-Loss only cares about
the data within the boundary, that is, it only penalizes the data
within the boundary, and no longer penalizes the data outside the boundary.
From a mathematical point of view, $y(w^T x + b) \geq 1$, the loss is 0.

----
### 2.

###### (a)
Prove: (a,b) = (0,1)<br>
Hint:
$\begin{array}{c} \mathbb{I}\{y^{(i)} = 1\} =
\begin{cases}
1 & \text{if } y^{(i)} = 1 \\
0 & \text{else}
\end{cases} \end{array}$<br>
$$\begin{align*}

\text{1. We Use log likehood} \\
\\
ℓ(θ) = \log L(\theta) = \sum_{i=1}^{m}y^{(i)}h(x^{(i)}) + (1-y^{(i)})\log(1-h(x^{(i)})) \\
\\
\text{2. We Set Gradient equal 0} \\
\\
\frac{\partial ℓ(θ)}{\partial \theta_j} = \sum_{i=1}^{m}(y^{(i)} - h(x^{(i)}))x_j^{(i)} = 0 \\
\\
\text{3. Set j = 0, $x_j^{i} = 1$, So We can get} \\
\\
\sum_{i=1}^{m}(y^{(i)} - h(x^{(i)}))x_j^{(i)} = 0 => \sum_{i=1}^{m}y^{(i)} = \sum_{i=1}^{m}h(x^{(i)}) \\
\\

h(x^{(i)}) = P(y^{(i)} = 1 | x^{(i)}; \theta) \\
y^{(i)} = \mathbb{I}\{y^{(i)} = 1\} \\
\\
\sum_{i=1}^{m} P(y^{(i)} = 1 | x^{(i)}; \theta) = \sum_{i=1}^{m}\mathbb{I}\{y^{(i)} = 1\} \\
\\

\text{4. When (a, b) = (0, 1), $I_{a,b} = \{x^{(i)}, y^{(i)}\}_{i=1}^{m}\ and \ |\{i \in I_{a,b}\}|$} = m \\
\\
\frac{\sum_{i \in I_{a,b}} P(y^{(i)} = 1|x^{(i)}; \theta)}{|\{{i \in I_{a,b}}\}|} = \frac{\sum_{i \in I_{a,b}} \mathbb{I}{\{y^{(i)} = 1}\}}{|\{{i \in I_{a,b}}\}|} \\
\\
\text{Too End!!!}

\end{align*}$$

##### (b)


The model is perfectly calibrated doesn't necessarily imply that the model achieves perfect accuracy.<br>
The converse is also not necessarily true.
Assume that $(a, b) = (0.5, 1)$<br>
When the model achieves perfect accuracy, the predictions are all correct.<br>
$$
\begin{align*}

\sum_{i \in I_{a,b}} \mathbb{I}\{y^{(i)} = 1\} = |\{i \in I_{a,b}\}|
\\
\text{For all $i \in I_{a,b}$} \\
0.5 < P(y^{(i)} = 1|x^{(i)}, \theta) < 1 \\
\\
So, We\ get \\
\frac{\sum_{i \in I_{a,b}} P(y^{(i)} = 1|x^{(i)}, \theta)}{|\{i \in I_{a,b}\}|} < \frac{\sum_{i \in I_{a,b}} \mathbb{I}\{y^{(i)} = 1\}}{|\{i \in I_{a,b}\}|}
\\
\\
\text{However, when the model is perfectly calibrated, the following property always hold} \\
\\
\frac{\sum_{i \in I_{a,b}} P(y^{(i)} = 1|x^{(i)}, \theta)}{|\{i \in I_{a,b}\}|} = \frac{\sum_{i \in I_{a,b}} \mathbb{I}\{y^{(i)} = 1\}}{|\{i \in I_{a,b}\}|} \\

\end{align*}
$$

So model is perfectly calibrated doesn't mean model achieves perfect accuracy. The converse neither.

##### (c)

When adding $L_2$ regularization, $\theta$ is not the maximum likelihood parameter learned after training.<br>
Furthermore, the loss function is
$$
\begin{align*}

J(\theta) = -\sum_{i=1}^{m} y^{(i)} \log h(x^{(i)}) + (1 - y^{(i)}) \log(1 - h(x^{(i)})) + \frac{\lambda}{2} \|\theta\|_2^2 \\

\text{After training, the gradients are equal to 0}
\\
\frac{\partial J(\theta)}{\partial \theta_j} = \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) x_j^{(i)} + \lambda \theta_j = 0
\\
\text{Set $j = 0$. Because $x_0^{(i)} = 1$, so}
\\
\sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) + \lambda \theta_0 = 0
\\
\sum_{i=1}^{m} h(x^{(i)}) + \lambda \theta_0 = \sum_{i=1}^{m} y^{(i)}
\\
\sum_{i=1}^{m} P(y^{(i)} = 1|x^{(i)}; \theta) + \lambda \theta_0 = \sum_{i=1}^{m} \mathbb{I}\{y^{(i)} = 1\} \\

\end{align*}
$$

So the model will not be well-calibrated.

---
### 3.

##### (a)
Prove: $$\theta_{MAP} = \arg\max_\theta p(y|x,\theta)p(\theta)$$

$$
\begin{align*}

\text{We need use Conditional Probability $p(A | B) = \frac{p(A, B)}{p(B)}$} \\
\text{Conditional and Joint Probability $p(x, y, \theta) = p(y | x, \theta) p(\theta | x) p(x)$} \\
\\
p(y|x,\theta)
= \frac{p(y, x, \theta)}{p(x, y)} 
= \frac{p(y|x, \theta)p(\theta|x)p(x)}{p(x, y)} \\
\\
\text{Asssume $p(\theta) = p(\theta|x)$} \\
\\
= \frac{p(y|x, \theta)p(\theta)p(x)}{p(x, y)} \\
= p(y|x, \theta)p(\theta) \cdot \frac{p(x)}{p(x, y)} \\
\\
We \ substitute \ this \ formula\ \theta_{MAP} = \arg\max_\theta p(y|x,\theta)p(\theta) \\
\\
\theta_{MAP} = \arg\max_\theta p(y|x,\theta)p(\theta) \cdot \frac{p(x)}{p(x, y)} \\
\end{align*}
$$
**We Know, When we find the maximum value of $\theta$, parameters unrelated to $\theta$ can be removed without affecting the optimization process.**<br>
**So, We Get** $$\theta_{MAP} = \arg\max_\theta p(y|x,\theta)p(\theta)$$

##### (b)
Prove: $$\theta_{MAP} = \arg\min_\theta -\log p(y|x,\theta) + \lambda||\theta||_2^{2}$$
and $\lambda$?

$$
\begin{align*}

\arg\max_\theta p(y|x, \theta)p(\theta) \\
= \arg\max_\theta \log p(y|x, \theta)p(\theta) \\
= \arg\max_\theta \log p(y|x, \theta) + \log p(\theta) \\
\text{We take the opposite, Get} \\
= -\arg\min_\theta \log p(y|x, \theta) - \log p(\theta) \\
\\
\text{We know $\theta \sim \mathcal{N}(0, \eta^2 I)$ \ is equivalent to use $L_2$ Regularization.} \\
\\
p(\theta) = \frac{1}{(2\pi)^{n/2}\eta^{n}}\exp(-\frac{1}{2\eta^{2}}\theta^{T}\theta) = (2\pi)^{-n/2}\eta^{-n}\exp\{-\frac{1}{2\eta^{2}}||\theta||_2^{2}\} \\
\\
\log p(\theta) = -\frac{n}{2}\log(2\pi) - n\log \eta - \frac{1}{2\eta^{2}}||\theta||_2^{2} \\
\\
\text{Drop $-\frac{n}{2}\log(2\pi) - n\log \eta$, It has nothing to do with Regularization.} \\
\\
\theta_{MAP} = \arg\min_\theta -\log p(y|x,\theta) + \frac{1}{2\eta^{2}}||\theta||_2^{2} \\

\lambda = \frac{1}{2\eta^{2}}

\end{align*}
$$

##### (c)

$$
\begin{align*}

\epsilon^{(i)} &\sim \mathcal{N}(0, \sigma^2) \\

y^{(i)} &= \theta^T x^{(i)} + \epsilon^{(i)} \\

y^{(i)} | x^{(i)}, \theta &\sim \mathcal{N}(\theta^T x^{(i)}, \sigma^2) \\

p(y^{(i)} | x^{(i)}, \theta) &= \frac{1}{\sqrt{2\pi}\sigma} \exp\left\{ -\frac{1}{2\sigma^2} \left(y^{(i)} - \theta^T x^{(i)}\right)^2 \right\} \\

p(\vec{y} | X, \theta) &= \prod_{i=1}^m p(y^{(i)} | x^{(i)}, \theta) \\

&= \prod_{i=1}^m \frac{1}{\sqrt{2\pi}\sigma} \exp\left\{ -\frac{1}{2\sigma^2} \left(y^{(i)} - \theta^T x^{(i)}\right)^2 \right\} \\

&= \frac{1}{(2\pi)^{m/2} \sigma^m} \exp\left\{ -\frac{1}{2\sigma^2} \sum_{i=1}^m \left(y^{(i)} - \theta^T x^{(i)}\right)^2 \right\} \\

&= \frac{1}{(2\pi)^{m/2} \sigma^m} \exp\left\{ -\frac{1}{2\sigma^2} \|X\theta - \vec{y}\|_2^2 \right\} \\
\\
\log p(\vec{y} | X, \theta) = -\frac{m}{2}\log(2\pi) - m\log\sigma - \frac{1}{2\sigma^2} \|X\theta - \vec{y}\|_2^2 \\

\theta_{MAP} = \arg\min_\theta \frac{1}{2\sigma^2} \|X\theta - \vec{y}\|_2^2 + \frac{1}{2\eta^{2}}||\theta||_2^{2} \\

J(\theta) = \frac{1}{2\sigma^2} \|X\theta - \vec{y}\|_2^2 + \frac{1}{2\eta^{2}}||\theta||_2^{2} \\
\\

\text{Data Item}\ f(\theta)=\frac{1}{2\sigma^2}\bigl(\vec y - X\theta\bigr)^\top\bigl(\vec y - X\theta\bigr) \\
\\
\text{Let $r = \vec y - X\theta$ => $\nabla_\theta r = -X$} \\
\\
\nabla_\theta u 
= \Bigl(\frac{\partial r}{\partial \theta}\Bigr)^\top 
  \frac{\partial u}{\partial r}
= (-X)^\top \,(2\,r)\\
\\
\nabla_\theta f
= \frac{1}{2\sigma^2}\bigl(2\,(-X)^\top r \bigr)
= -\frac{1}{\sigma^2}X^\top(\vec y - X\theta)
= \frac{1}{\sigma^2}\bigl(X^\top X\theta - X^\top\vec y\bigr) \\
\\
\text{Regularization Item} \ g(\theta)=\frac{1}{2\eta^2}\,\theta^\top\theta,
\qquad
\nabla_\theta g
= \frac{1}{2\eta^2}\,2\theta
= \frac{1}{\eta^2}\,\theta \\
\\
\nabla_\theta J(\theta)
=\nabla_\theta f + \nabla_\theta g
=\frac{1}{\sigma^2}(X^\top X\theta - X^\top\vec y)
\;+\;\frac{1}{\eta^2}\,\theta \\
\\
\text{Set to 0} \\
\\
\frac{1}{\sigma^2}(X^\top X\theta - X^\top\vec y) + \frac{1}{\eta^2}\theta \;=\;0 \\
\\
\frac{1}{\sigma^2}(X^\top X\theta - X^\top\vec y)
\;+\;\frac{1}{\eta^2}\theta = 0
\;\;\Longrightarrow\;\;
X^\top X\theta - X^\top\vec y + \frac{\sigma^2}{\eta^2}\,\theta = 0 \\
\\
\text{Merge} \\
\\
\bigl(X^\top X + \frac{\sigma^2}{\eta^2}I\bigr)\,\theta
= X^\top\vec y \\
\\
\text{Multiply the inverse matrix} \\
\\
\boxed{\theta_{\rm MAP} =\arg\min_\theta J(\theta) = \bigl(X^\top X + \tfrac{\sigma^2}{\eta^2}I\bigr)^{-1}X^\top\vec y.} \\

\end{align*}
$$

##### (d)
Solution
$$
\begin{align*}

\theta \sim \mathcal{L}(0, bI) \\

p(\theta) = \frac{1}{(2b)^n} \exp\left\{-\frac{1}{b}||\theta||_1\right\} \\

\log p(\theta) = -n \log(2b) - \frac{1}{b}||\theta||_1 \\

\theta_{\text{MAP}} = \arg \min_\theta \frac{1}{2\sigma^2} ||X\theta - \vec{y}||_2^2 - \log p(\theta) \\

= \arg \min_\theta \frac{1}{2\sigma^2} ||X\theta - \vec{y}||_2^2 + \frac{1}{b}||\theta||_1 \\

J(\theta) = ||X\theta - \vec{y}||_2^2 + \gamma||\theta||_1 \\

\theta_{\text{MAP}} = \arg \min_\theta J(\theta) \\

\gamma = \frac{2\sigma^2}{b}

\end{align*}
$$

### 4.

##### (a)
