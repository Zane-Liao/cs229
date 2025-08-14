# PS2 Solution

## 1. Logistic Regression: Training stability

### (a) What is the most notable diﬀerence in training the logistic regression model on datasets A and B? 

- Data Sets A Converge quickly, Data Sets B No convergence.

### (b) Investigate why the training procedure behaves unexpectedly on dataset B, but not on A. Provide hard evidence

- Dataset A is linearly inseparable and has a unique minimum value,but data set B is linearly separable.
In mathematics, when a linearlyseparable data set appears without adding a regularization term,
the logarithmic loss function of logistic regression has no minimum value,
which will cause non-convergence.
$\mathcal{L}(\theta) = -\sum_{i=1}^{n} \left[y_i \log(p_i) + (1 - y_i) \log(1 - p_i)\right]$

### (c) State whether or not it would lead to the provided training algorithm converging on datasets such as B

**i. Using a diﬀerent constant learning rate**<br>
**ii. Decreasing the learning rate over time**<br>
**iii. Linear scaling of the input features**<br>
**iv. Adding a regularization term $||\theta||^{2}_2$ to the loss function**<br>
**v. Adding zero-mean Gaussian noise to the training data or labels**


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

### (d) Are support vector machines, which use the hinge loss, vulnerable to datasets like B? Why or why not?

- It is not easily affected because Hinge-Loss only cares about
the data within the boundary, that is, it only penalizes the data
within the boundary, and no longer penalizes the data outside the boundary.
From a mathematical point of view, $y(w^T x + b) \geq 1$, the loss is 0.

## 2. Model Calibration

### (a) According to the question, Show that the above property holds true for the described logistic regression model over the range (a,b) = (0,1)

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

### (b) If we have a binary classification model that is perfectly calibrated—that is, the property we just proved holds for any (a,b) ⊂[0,1]—does this necessarily imply that the model achieves perfect accuracy?

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

### (c) Discuss what eﬀect including L2 regularization in the logistic regression objective has on model calibration

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

## 3. Bayesian Interpretation of Regularization

### (a) Prove: $\theta_{MAP} = \arg\max_\theta p(y|x,\theta)p(\theta)$

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

### (b) Prove: $\theta_{MAP} = \arg\min_\theta -\log p(y|x,\theta) + \lambda||\theta||_2^{2}$ and $\lambda$?

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

### (c) Come up with a closed form expression for $θ_{MAP}$

$$
\begin{aligned}

& \epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2) \\

& y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)} \\

& y^{(i)} | x^{(i)}, \theta \sim \mathcal{N}(\theta^T x^{(i)}, \sigma^2) \\

& p(y^{(i)} | x^{(i)}, \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left\{ -\frac{1}{2\sigma^2} \left(y^{(i)} - \theta^T x^{(i)}\right)^2 \right\} \\

& p(\vec{y} | X, \theta) = \prod_{i=1}^m p(y^{(i)} | x^{(i)}, \theta) \\

&= \prod_{i=1}^m \frac{1}{\sqrt{2\pi}\sigma} \exp\left\{ -\frac{1}{2\sigma^2} \left(y^{(i)} - \theta^T x^{(i)}\right)^2 \right\} \\

&= \frac{1}{(2\pi)^{m/2} \sigma^m} \exp\left\{ -\frac{1}{2\sigma^2} \sum_{i=1}^m \left(y^{(i)} - \theta^T x^{(i)}\right)^2 \right\} \\

&= \frac{1}{(2\pi)^{m/2} \sigma^m} \exp\left\{ -\frac{1}{2\sigma^2} \|X\theta - \vec{y}\|_2^2 \right\} \\
\\

& \log p(\vec{y} | X, \theta) = -\frac{m}{2}\log(2\pi) - m\log\sigma - \frac{1}{2\sigma^2} \|X\theta - \vec{y}\|_2^2 \\

& \theta_{MAP} = \arg\min_\theta \frac{1}{2\sigma^2} \|X\theta - \vec{y}\|_2^2 + \frac{1}{2\eta^{2}}||\theta||_2^{2} \\

& J(\theta) = \frac{1}{2\sigma^2} \|X\theta - \vec{y}\|_2^2 + \frac{1}{2\eta^{2}}||\theta||_2^{2} \\
\\

& \text{Data Item}\ f(\theta)=\frac{1}{2\sigma^2}\bigl(\vec y - X\theta\bigr)^\top\bigl(\vec y - X\theta\bigr) \\
\\
& \text{Let $r = \vec y - X\theta$ => $\nabla_\theta r = -X$} \\
\\
& \nabla_\theta u 
= \Bigl(\frac{\partial r}{\partial \theta}\Bigr)^\top 
  \frac{\partial u}{\partial r}
= (-X)^\top \,(2\,r)\\
\\
& \nabla_\theta f
= \frac{1}{2\sigma^2}\bigl(2\,(-X)^\top r \bigr)
= -\frac{1}{\sigma^2}X^\top(\vec y - X\theta)
= \frac{1}{\sigma^2}\bigl(X^\top X\theta - X^\top\vec y\bigr) \\
\\
& \text{Regularization Item} \ g(\theta)=\frac{1}{2\eta^2}\,\theta^\top\theta,
\qquad
\nabla_\theta g
= \frac{1}{2\eta^2}\,2\theta
= \frac{1}{\eta^2}\,\theta \\
\\
& \nabla_\theta J(\theta)
=\nabla_\theta f + \nabla_\theta g
=\frac{1}{\sigma^2}(X^\top X\theta - X^\top\vec y)
\;+\;\frac{1}{\eta^2}\,\theta \\
\\
& \text{Set to 0} \\
\\
& \frac{1}{\sigma^2}(X^\top X\theta - X^\top\vec y) + \frac{1}{\eta^2}\theta \;=\;0 \\
\\
& \frac{1}{\sigma^2}(X^\top X\theta - X^\top\vec y)
\;+\;\frac{1}{\eta^2}\theta = 0
\;\;\Longrightarrow\;\;
X^\top X\theta - X^\top\vec y + \frac{\sigma^2}{\eta^2}\,\theta = 0 \\
\\
& \text{Merge} \\
\\
& \bigl(X^\top X + \frac{\sigma^2}{\eta^2}I\bigr)\,\theta
= X^\top\vec y \\
\\
& \text{Multiply the inverse matrix} \\
\\
& \boxed{\theta_{\rm MAP} =\arg\min_\theta J(\theta) = \bigl(X^\top X + \tfrac{\sigma^2}{\eta^2}I\bigr)^{-1}X^\top\vec y.} \\

\end{aligned}
$$

### (d) Prove: $\theta_{MAP}$ is equivalent to the solution of linear regression with $L_1$ regularization, where the loss is specified as: $J(\theta) = ||X\theta - \vec{y}||_2^2 + \gamma||\theta||_1$ Also, what is the value of $\gamma$?

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

## 4. Constructing kernel function

**For each of the functions K below, state whether it is necessarily a kernel. If you think it is, prove it; if you think it isn’t, give a counter-example**

### (a) $K(x,z) = K_1(x,z) + K_2(x,z)$

$$
\begin{align*}
\text{Yes, $K_1$ and $K_2$ are both PSD, so $K_1 + K_2$ is PSD.} \\
z^T K z = z^T (K_1 + K_2) z = z^T K_1 z + z^T K_2 z \geq 0

\end{align*}
$$

### (b) $K(x,z) = K_1(x,z) - K_2(x,z)$

$$
\begin{align*}
\text{No, although $K_1$ and $K_2$ are both PSD, $K_1 - K_2$ may not be PSD.
For example, $K_2 = 2K_1$} \\
z^T K z = z^T (K_1 - K_2) z = z^T (K_1 - 2K_1) z = -z^T K_1 z \leq 0

\end{align*}
$$

### (c) $K(x,z) = aK_1(x,z)$

$$
\begin{align*}
\text{Yes, $K_1$ is PSD, so $aK_1$ $(a \in \mathbb{R}^+)$ is PSD.} \\
z^T K z = z^T a K_1 z = a \cdot z^T K_1 z \geq 0

\end{align*}
$$

### (d) $K(x,z) = -aK_1(x,z)$

$$
\begin{align*}
\text{No, $K_1$ is PSD, so $-aK_1$ $(a \in \mathbb{R}^+)$ is not PSD.} \\
z^T K z = z^T (-a K_1) z = -a \cdot z^T K_1 z \leq 0

\end{align*}
$$

### (e) $K(x,z) = K_1(x,z)K_2(x,z)s$

$$
\begin{align*}
\text{Yes, $K_1 K_2$ is PSD.} \\
z^T K z &= \sum \sum z_i K_{ij} z_j \\
&= \sum \sum z_i K_1 \left(x^{(i)}, x^{(j)}\right) K_2 \left(x^{(i)}, x^{(j)}\right) z_j \\
&= \sum \sum z_i \phi_1(x^{(i)})^T \phi_1(x^{(j)}) \phi_2(x^{(i)})^T \phi_2(x^{(j)}) z_j \\
&= \sum \sum z_i \sum_a \phi_{1a}(x^{(i)}) \phi_{1a}(x^{(j)}) \sum_b \phi_{2b}(x^{(i)}) \phi_{2b}(x^{(j)}) z_j \\
&= \sum \sum \sum \sum z_i \phi_{1a}(x^{(i)}) \phi_{1a}(x^{(j)}) \phi_{2b}(x^{(i)}) \phi_{2b}(x^{(j)}) z_j \\
&= \sum \sum \sum \left(z_i \phi_{1a}(x^{(i)}) \phi_{2b}(x^{(i)})\right)^2 \geq 0
\end{align*}
$$

### (f) $K(x,z) = f(x)f(z)$

$$
\begin{align*}
\text{Yes, $K$ is PSD.} \\
\text{$f : \mathbb{R}^n \mapsto \mathbb{R}$ is a real-valued function, then} \\
z^T K z &= \sum \sum z_i K_{ij} z_j \\
&= \sum \sum z_i f(x^{(i)}) f(x^{(j)}) z_j \\
&= \sum \left(z_i f(x^{(i)})\right)^2 \geq 0

\end{align*}
$$

### (g) $K(x,z) = K_3(\phi(x), \phi(z))$

$$
\begin{align*}
\text{Yes, $K_3(\phi(x), \phi(z))$ is a valid kernel, no matter what the inputs are.} \\
\\
\text{Yes, $p(K_1)$ is a valid kernel.} \\
\\
\text{$p(x)$ is a polynomial function with coefficients $c_k > 0$, $k = 0, 1, \ldots, n$} \\
\\
p(x) = \sum_{k=0}^n c_k x^k \\
\\
K(x, z) = p(K_1(x, z)) = \sum_{k=0}^n c_k \left(K_1(x, z)\right)^k
\end{align*}
$$

### (h) $K(x,z) = p(K_1(x,z))$

$$
\begin{aligned}

& \text{From (e) we know $K(x, z) = K_1(x, z) K_2(x, z)$ is a valid kernel, so $K(x, z) = \left(K_1(x, z)\right)^k$ is valid.} \\
\\
& \text{From (a) and (c), we know $K(x, z) = K_1(x, z) + K_2(x, z)$ and $K(x, z) = a K_1(x, z)$, $a \in \mathbb{R}^+$ are both valid.} \\
\\
& \text{So $K(x, z) = \sum_{k=0}^n c_k \left(K_1(x, z)\right)^k$ is a valid kernel.} \\

\end{aligned}
$$

## 5. Kernelization of Perceptron

### (a) How would you (implicitly) represent the high-dimensional parameter vector $\theta^{(i)}$, including the initial value $\theta^{(0)} = 0$?

$$
\begin{aligned}

& \text{i} \\
& \text{Perceptron Algorithm Update \ $\theta^{(t)} \;=\; \theta^{(t-1)}
\;+\;\alpha\bigl(y^{(t)} - h_{\theta^{(t-1)}}(x^{(t)})\bigr)\,\phi\bigl(x^{(t)}\bigr)$} \\
\\
& \text{We use \ $\beta_t \;=\; \alpha\bigl(y^{(t)} - h_{\theta^{(t-1)}}(x^{(t)})\bigr)$} \\
\\
& \text{So, \ $\theta^{(t)} = \theta^{(t-1)} + \beta_t\,\phi\bigl(x^{(t)}\bigr)$} \\
\\
& \text{We know $\theta^{(0)} = \vec 0$} \\
\\
& \sum_{j=1}^0 \beta_j\,\phi\bigl(x^{(j)}\bigr)
= 0 \\
\\
& \text{Assume that for some i - 1 >= 0, We have }\\
\\
& \theta^{(i-1)} \;=\; \sum_{j=1}^{i-1} \beta_j\,\phi\bigl(x^{(j)}\bigr) \\
\\
& \text{We know \ $\theta^{(i)}
= \theta^{(i-1)} + \beta_i\,\phi\bigl(x^{(i)}\bigr)$} \\
\\
& \theta^{(i)}
= \Bigl(\sum_{j=1}^{i-1} \beta_j\,\phi\bigl(x^{(j)}\bigr)\Bigr)
  + \beta_i\,\phi\bigl(x^{(i)}\bigr)
= \sum_{j=1}^{i} \beta_j\,\phi\bigl(x^{(j)}\bigr) \\
\\
& \text{ii} \\
\\
& h_{\theta^{(i)}}\left(\phi\left(x^{(i+1)}\right)\right) = g\left(\theta^{(i)T} \phi\left(x^{(i+1)}\right)\right) \\
&= \text{sign}\left(\theta^{(i)T} \phi\left(x^{(i+1)}\right)\right) \\
&= \text{sign}\left(\sum_{j=1}^{i} \beta_j \phi\left(x^{(j)}\right)^T \phi\left(x^{(i+1)}\right)\right) \\
&= \text{sign}\left(\sum_{j=1}^{i} \beta_j \left\langle \phi\left(x^{(j)}\right), \phi\left(x^{(i+1)}\right) \right\rangle\right) \\
&= \text{sign}\left(\sum_{j=1}^{i} \beta_j K\left(x^{(j)}, x^{(i+1)}\right)\right)
\\
& \text{iii} \\
\\
& \theta^{(i+1)} := \theta^{(i)} + \alpha \left(y^{(i+1)} - h_{\theta^{(i)}}\left(\phi\left(x^{(i+1)}\right)\right)\right) \phi\left(x^{(i+1)}\right) 
\\
&= \sum_{j=1}^{i} \beta_j \phi\left(x^{(j)}\right) + \alpha \left(y^{(i+1)} - \text{sign}\left(\sum_{j=1}^{i} \beta_j K\left(x^{(j)}, x^{(i+1)}\right)\right)\right) \phi\left(x^{(i+1)}\right) \\

& \beta_{i+1} = \alpha \left(y^{(i+1)} - \text{sign}\left(\sum_{j=1}^{i} \beta_j K\left(x^{(j)}, x^{(i+1)}\right)\right)\right)

\end{aligned}
$$

### (b) (c) Code Problem...

The dot product kernel is only effective for linear models.

## 6. Sorting spam mail

**use the naive Bayes algorithm and an SVM to build a spam classifier**

### (a) (b) (c) (d) Code Problem...

### (b) Find a way to compute Naive Bayes’ predicted class labels without explicitly representing very small numbers such as p(x|y) [Hint: Think about using logarithms]

$$
\begin{align*}

p(y = 1|x) = \frac{\prod_{j=1}^{d} p(x_j|y = 1)p(y = 1)}{\prod_{j=1}^{d} p(x_j|y = 1)p(y = 1) + \prod_{j=1}^{d} p(x_j|y = 0)p(y = 0)} \\
\\
= \frac{1}{1 + \frac{\prod_{j=1}^{d} p(x_j|y=0)p(y=0)}{\prod_{j=1}^{d} p(x_j|y=1)p(y=1)}}\\
\\
p(y = 1|x) > 0.5 \\
\\
\prod_{j=1}^{d} p(x_j|y = 1)p(y = 1) > \prod_{j=1}^{d} p(x_j|y = 0)p(y = 0) \\
\\
\log \left(\prod_{j=1}^{d} p(x_j|y = 1)p(y = 1)\right) > \log \left(\prod_{j=1}^{d} p(x_j|y = 0)p(y = 0)\right) \\
\\
\sum_{j=1}^{d} \log p(x_j|y = 1) + \log p(y = 1) > \sum_{j=1}^{d} \log p(x_j|y = 0) + \log p(y = 0)
\end{align*}
$$