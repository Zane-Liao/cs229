
### 1. Linear Classifiers (logistic regression and GDA)
(a)
$$\begin{align*}

\frac{\partial J(\theta)}{\partial \theta_j} &= -\frac{1}{m} \sum_{i=1}^{m} y^{(i)} \frac{g(\theta^T x^{(i)})[1 - g(\theta^T x^{(i)})]}{g(\theta^T x^{(i)})} x_j^{(i)} - (1 - y^{(i)}) \frac{g(\theta^T x^{(i)})[1 - g(\theta^T x^{(i)})]}{1 - g(\theta^T x^{(i)})} x_j^{(i)} \tag{1.1.1} \\

&= -\frac{1}{m} \sum_{i=1}^{m} y^{(i)} [1 - g(\theta^T x^{(i)})] x_j^{(i)} - (1 - y^{(i)}) g(\theta^T x^{(i)}) x_j^{(i)} \tag{1.1.2} \\

&= \frac{1}{m} \sum_{i=1}^{m} [g(\theta^T x^{(i)}) - y^{(i)}] x_j^{(i)} \tag{1.1.3}

\end{align*}$$
$$\nabla_\theta J(\theta) = \frac{1}{m} X^T (g(X\theta) - Y)$$

$$H_{jk} = \frac{\partial^2 J(\theta)}{\partial \theta_j \partial \theta_k} = \frac{1}{m} \sum_{i=1}^{m} g(\theta^T x^{(i)}) [1 - g(\theta^T x^{(i)})] x_j^{(i)} x_k^{(i)}$$

$$H = \frac{1}{m} [X^T \cdot g(X\theta) \cdot (1 - g(X\theta))] X$$

$$z^T H z = \frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{n} \sum_{k=1}^{n} g(\theta^T x^{(i)}) [1 - g(\theta^T x^{(i)})] x_j^{(i)} x_k^{(i)} z_j z_k$$

$$= \frac{1}{m} \sum_{i=1}^{m} g(\theta^T x^{(i)}) [1 - g(\theta^T x^{(i)})] [(x^{(i)})^T z]^2 \geq 0$$
(b) Code Problem...
(c)
$$
\begin{align*}

{p(y=1|x)}
&= \frac{p(x|y=1)p(y=1)}{p(x|y=1)p(y=1) +p(x|y=0)p(y=0)} \tag{1.2.1} \\

&= \frac{\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp(-\frac{1}{2}(x - \mu_1)^{T}\Sigma^{-1}(x - \mu_1)\cdot \phi^{1} \cdot (1-\phi)^{1-1} }{\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp(-\frac{1}{2}(x - \mu_1)^{T}\Sigma^{-1}(x - \mu_1)\cdot \phi^{1} \cdot (1-\phi)^{1-1} + \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp(-\frac{1}{2}(x - \mu_0)^{T}\Sigma^{-1}(x - \mu_0)\cdot \phi^{0} \cdot (1-\phi)^{1-0}} \tag{1.2.2} \\

&= \frac{\exp(-\frac{1}{2}(x - \mu_1)^{T}\Sigma^{-1}(x - \mu_1)) \cdot \phi}{\exp(-\frac{1}{2}(x - \mu_1)^{T}\Sigma^{-1}(x - \mu_1)) \cdot \phi + \exp(-\frac{1}{2}(x - \mu_0)^{T}\Sigma^{-1}(x - \mu_0)) \cdot (1 - \phi)} \tag{1.2.3} \\

&= \frac{1}{1 + \frac{\exp(-\frac{1}{2}(x - \mu_1)^{T}\Sigma^{-1}(x - \mu_1)) \cdot (1 - \phi)}{\exp(-\frac{1}{2}(x - \mu_1)^{T}\Sigma^{-1}(x - \mu_1)) \cdot \phi}} \tag{1.2.4} \\

&= \frac{1}{1 + \exp\{-\frac{1}{2}(x - \mu_1)^{T}\Sigma^{-1}(x - \mu_1) -\frac{1}{2}(x - \mu_0)^{T}\Sigma^{-1}(x - \mu_0)\}} \tag{1.2.5} \\

&= \frac{1}{1 + \exp\{-[(\Sigma^{-1}(\mu_1 - \mu_0))^{T}x + \frac{1}{2}(\mu_0 + \mu_1)^{T}\Sigma^{-1}(\mu_0 - \mu_1) - \ln(\frac{1 - \phi}{\phi})]\}} \tag{1.2.6}

\end{align*}

$$
(1.2.5) => (1.2.6)
$$
\begin{align*}

A = \frac{1}{2}(x - \mu_1)^{T}\Sigma^{-1}(x - \mu_1) - \frac{1}{2}(x - \mu_0)^{T}\Sigma^{-1}(x - \mu_0) \\

(x - \mu_1)^{T}\Sigma^{-1}(x - \mu_1) = x^{T}\Sigma^{-1}x - x^{T}\Sigma^{-1}\mu_1 - \mu_1^{T}\Sigma^{-1}x + \mu_1^{T}\Sigma^{-1}\mu_1 \\

(x - \mu_0)^{T}\Sigma^{-1}(x - \mu_0) = x^{T}\Sigma^{-1}x - x^{T}\Sigma^{-1}\mu_0 - \mu_0^{T}\Sigma^{-1}x + \mu_0^{T}\Sigma^{-1}\mu_1

\end{align*}
$$
Because $\Sigma^{-1}$ With Symmetric Matrix, So 
$$
\Sigma^{-1}=>X^{T}\Sigma^{-1}\mu_k = \mu_k^{T}\Sigma^{-1}
$$
$$
\begin{align*}

(x - \mu_1)^{T}\Sigma^{-1}(x - \mu_1) - (x - \mu_0)^{T}\Sigma^{-1}(x - \mu_0)

&= [x^{T}\Sigma^{-1}x - x^{T}\Sigma^{-1}\mu_1 - \mu_1^{T}\Sigma^{-1}x + \mu_1^{T}\Sigma^{-1}\mu_1] - [x^{T}\Sigma^{-1}x - x^{T}\Sigma^{-1}\mu_0 - \mu_0^{T}\Sigma^{-1}x + \mu_0^{T}\Sigma^{-1}\mu_1] \\

&= [x^{T}\Sigma^{-1}x - x^{T}\Sigma^{-1}x] + [(-x^{T}\Sigma^{-1}\mu_1 - \mu_1^{T}\Sigma^{-1}x) - (-x^{T}\Sigma^{-1}\mu_0 - \mu_0^{T}\Sigma^{-1}x)]
+ [\mu_1^{T}\Sigma^{-1}\mu_1 - \mu_0^{T}\Sigma^{-1}\mu_1] \\

&= (-x^{T}\Sigma^{-1}\mu_1 - \mu_1^{T}\Sigma^{-1}x) - (x^{T}\Sigma^{-1}\mu_0 - \mu_0^{T}\Sigma^{-1}x) \\

&= -(x^{T}\Sigma^{-1}\mu_1 + \mu_1^{T}\Sigma^{-1}x) + (x^{T}\Sigma^{-1}\mu_0 + \mu_0^{T}\Sigma^{-1}x) \\

&= -2\mu_1^{T}\Sigma^{-1}x + 2\mu_0^{T}\Sigma^{-1}x \\

&= [2(\mu_0- \mu_1)^{T}\Sigma^{-1}x + (\mu_1^{T}\Sigma^{-1}\mu_1 - \mu_0^{T}\Sigma^{-1}\mu_0)] \times \frac{1}{2}  \\

&= (\mu_0 - \mu_1)^{T}\Sigma^{-1}x + \frac{1}{2}(\mu_1^{T}\Sigma^{-1}\mu_1 - \mu_0^{T}\Sigma^{-1}\mu_0) \\

&= -(\mu_1 - \mu_0)^{T}\Sigma^{-1}x + \frac{1}{2}(\mu_1^{T}\Sigma^{-1}\mu_1 - \mu_0^{T}\Sigma^{-1}\mu_0) \\

&= [(\Sigma^{-1}(\mu_1 - \mu_0)^{T}x + \frac{1}{2}(\mu_0 + \mu_1)^{T}\Sigma^{-1}(\mu_0 - \mu_1)] \\

\end{align*}
$$
(d)

$$
\begin{align*}
\mu_{y^{(i)}} &= \mathbf{1}\{y^{(i)} = 0\}\mu_0 + \mathbf{1}\{y^{(i)} = 1\}\mu_1 \\

p(x^{(i)}|y^{(i)}; \mu_0, \mu_1, \Sigma) &= \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left\{-\frac{1}{2}(x^{(i)} - \mu_{y^{(i)}})^T \Sigma^{-1} (x^{(i)} - \mu_{y^{(i)}})\right\} \\


p(y^{(i)}; \phi) &= \phi^{\mathbf{1}\{y^{(i)}=1\}} (1-\phi)^{1-\mathbf{1}\{y^{(i)}=1\}} \\


\ell &= \sum_{i=1}^m \log p(x^{(i)}|y^{(i)}; \mu_0, \mu_1, \Sigma) + \sum_{i=1}^m \log p(y^{(i)}; \phi) \\

&= \sum_{i=1}^m \log \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left\{-\frac{1}{2}(x^{(i)} - \mu_{y^{(i)}})^T \Sigma^{-1} (x^{(i)} - \mu_{y^{(i)}})\right\} + \sum_{i=1}^m \log \phi^{\mathbf{1}\{y^{(i)}=1\}} (1-\phi)^{1-\mathbf{1}\{y^{(i)}=1\}} \\

&= -\frac{mn}{2}\log(2\pi) - \frac{m}{2}\log|\Sigma| - \frac{1}{2}\sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}})^T \Sigma^{-1} (x^{(i)} - \mu_{y^{(i)}})\\
&\quad + \sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\} \log \phi + \left(m - \sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}\right) \log(1-\phi) \\

\frac{\partial \ell}{\partial \phi} &= \frac{1}{\phi} \sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\} + \frac{1}{\phi-1}\left(m - \sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}\right) \\


\frac{\partial \ell}{\partial \mu_{y^{(i)}}} &= \Sigma^{-1} \sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}}) \\


\frac{\partial \mu_{y^{(i)}}}{\partial \mu_0} &= \mathbf{1}\{y^{(i)} = 0\}, \quad \frac{\partial \mu_{y^{(i)}}}{\partial \mu_1} = \mathbf{1}\{y^{(i)} = 1\} \\


\frac{\partial \ell}{\partial \mu_0} &= \frac{\partial \ell}{\partial \mu_{y^{(i)}}} \frac{\partial \mu_{y^{(i)}}}{\partial \mu_0} = \Sigma^{-1} \sum_{i=1}^m \left(x^{(i)} \mathbf{1}\{y^{(i)} = 0\} - \mu_0 \mathbf{1}\{y^{(i)} = 0\}\right) \\



\frac{\partial \ell}{\partial \mu_1} &= \frac{\partial \ell}{\partial \mu_{y^{(i)}}} \frac{\partial \mu_{y^{(i)}}}{\partial \mu_1} = \Sigma^{-1} \sum_{i=1}^m \left(x^{(i)} \mathbf{1}\{y^{(i)} = 1\} - \mu_1 \mathbf{1}\{y^{(i)} = 1\}\right) \\


\frac{\partial \ell}{\partial \Sigma} &= -\frac{m}{2}\Sigma^{-1} + \frac{1}{2}\Sigma^{-1} \left(\sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T\right) \Sigma^{-1} \\

\begin{cases} \frac{\partial \ell}{\partial \phi} &= 0 \\ \frac{\partial \ell}{\partial \mu_0} &= 0 \\ \frac{\partial \ell}{\partial \mu_1} &= 0 \\ \frac{\partial \ell}{\partial \Sigma} &= 0 \end{cases} \Rightarrow \begin{cases} \phi &= \frac{1}{m} \sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\} \\ \mu_0 &= \frac{\sum_{i=1}^m \mathbf{1}\{y^{(i)}=0\}x^{(i)}}{\sum_{i=1}^m \mathbf{1}\{y^{(i)}=0\}} \\ \mu_1 &= \frac{\sum_{i=1}^m \mathbf{1}\{y^{(i)}=1\}x^{(i)}}{\sum_{i=1}^m \mathbf{1}\{y^{(i)}=1\}} \\ \Sigma &= \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T \end{cases}

\end{align*}
$$

### 2. Incomplete, Positive-Only Labels
(a)
$$ 

\begin{align*}
P(y = 1|t = 1, x)P(t = 1|x)P(x) &= P(y = 1, t = 1, x) = P(t = 1|y = 1, x)P(y = 1|x)P(x) \\
P(t = 1|x) &= P(y = 1|x) \frac{P(t = 1|y = 1, x)}{P(y = 1|t = 1, x)} \\
P(t = 1|y = 1, x) &= 1, \quad P(y = 1|t = 1, x) = P(y = 1|t = 1) \\
P(t = 1|x) &= \frac{P(y = 1|x)}{P(y = 1|t = 1)} \\
P(y = 1|t = 1) &= \alpha \\
\end{align*}$$
Another Solution:
$$
\begin{align*}

p(y^{(i)} \ \vert \ x^{(i)}) &= \sum_{t^{(i)}} p(y^{(i)} = 1, t^{(i)} \ \vert \ x^{(i)}) \\

&= p(y^{(i)} = 1, t^{(i)} = 1 \ \vert \ x^{(i)}) + p(y^{(i)} = 1, t^{(i)} = 0 \ \vert \ x^{(i)}) \\

&= p(y^{(i)} = 1, t^{(i)} = 1 \ \vert \ x^{(i)}) + 0 \\

&= p(y^{(i)} = 1 \ \vert \ t^{(i)} = 1, x^{(i)}) \ p(t^{(i)} = 1 \ \vert \ x^{(i)}) \\

&= p(y^{(i)} = 1 \ \vert \ t^{(i)} = 1) \ p(t^{(i)} = 1 \ \vert \ x^{(i)}) \\

&= \alpha \ p(t^{(i)} = 1 \ \vert \ x^{(i)})

\end{align*}
$$
$p(t^{(i)}=1|x^{(i)}) = \frac{p(y^{(i)}|x^{(i)})}{\alpha}$

(b)
$$
h(x) \approx p(y = 1|x) = p(t = 1|x)\alpha \approx \alpha \quad \text{for all } x \in V_+
$$
Another Solution:
$$
\begin{align*}

h(x^{(i)}) & \approx p(y^{(i)} = 1 \ \vert \ x^{(i)}) \\

&= \alpha \ p(t^{(i)} = 1 \ \vert \ x^{(i)}) \\

& \approx \alpha \cdot 1 \\

&= \alpha

\end{align*}
$$

(c)(d)(e) Code Problem...

### 3. Poisson Regression

(a)
$$
\begin{align*}

p(y; \ \lambda) &= \frac{e^{- \lambda} \ \lambda^y}{y \ !} \\

&= \frac{1}{y \ !} \exp (y \log \lambda - \lambda) \\

&= b(y) \exp(\eta^T T(y) - a(\eta))

\end{align*}
$$
where $b(y) = \frac{1}{y \ !}$, $T(y) = y$, $\eta = \log \lambda$ and $a(\eta) = \exp(\eta)$.

(b)
$$
\begin{align*}

\mathbb{E} [T(y); \ \eta] & = \mathbb{E} [y; \ \eta] \\

&= \lambda \\

&= \exp (\eta)

\end{align*}
$$

(c)

Recall the design choices of GLM:
1. $y \ \vert \ x; \ \theta \sim \mathrm{ExponentialFamily} (\eta)$
2. $h(x) = \mathbb{E} [y \ \vert \ x]$
3. $\eta = \theta^T x$

Plug them into $\ell$:
$$
\begin{align*}

\ell (\theta) & = \log p(y^{(i)} \ \vert \ x^{(i)}; \ \theta) \\

&= \log b(y^{(i)}) \exp (\eta^T T(y^{(i)}) - a(y^{(i)})) \\

&= (x^{(i)})^T \theta y^{(i)} - \exp (\theta^T x^{(i)}) - \log y^{(i)} !

\end{align*}
  $$
By design choice 2 and 3:

$$
\begin{align*}

h(x) & = \mathbb{E} [y \ \vert \ x] \\

&= \lambda \\

&= \exp (\eta) \\

&= \exp (\theta^T x)

\end{align*}
$$
Therefore,
$$
\begin{align*}

\frac{\partial}{\partial \theta}\ell (\theta) & = x^{(i)} y^{(i)} - x^{(i)} \exp (\theta^T x^{(i)}) \\

&= \big( y^{(i)} - h_\theta(x^{(i)}) \big) \ x^{(i)}

\end{align*}$$

The stochastic gradient ascent update rule is:
$$\theta := \theta + \alpha \ \big( y^{(i)} - h_\theta (x^{(i)}) \big) \ x^{(i)}$$

### 4. Convexity of Generalized Linear Models

(a)
Proof:
$$
\begin{align*}

\frac{\partial}{\partial \eta} p(y; \ \eta) &= \frac{\partial}{\partial \eta} \big( b(y) \exp (\eta y - a(\eta)) \big) \\

&= b(y) \exp (\eta y - a(\eta)) (y - \frac{\partial}{\partial \eta} a(\eta)) \\

&= y \ p(y; \ \eta) - p(y; \ \eta) \frac{\partial}{\partial \eta} a(\eta)

\end{align*}
$$
which indicates:
$$y \ p(y; \ \eta) = \frac{\partial}{\partial \eta} p(y; \ \eta) + p(y; \ \eta) \frac{\partial}{\partial \eta} a(\eta)$$
Therefore,
$$
\begin{align*}

\mathbb{E} [ \ Y; \ \eta \ ] & = \mathbb{E} [ \ Y \ \vert \ X; \ \eta \ ] \\

&= \int y \ p(y; \ \eta) \ dy \\

&= \int \frac{\partial}{\partial \eta} p(y; \ \eta) + p(y; \ \eta) \frac{\partial}{\partial \eta} a(\eta) \ dy \\

&= \int \frac{\partial}{\partial \eta} p(y; \ \eta) \ dy + \int p(y; \ \eta) \frac{\partial}{\partial \eta} a(\eta) \ dy \\

&= \frac{\partial}{\partial \eta} \int p(y; \ \eta) \ dy + \frac{\partial}{\partial \eta} a(\eta) \int p(y; \ \eta) \ dy \\

&= \frac{\partial}{\partial \eta} \cdot 1 + \frac{\partial}{\partial \eta} a(\eta) \cdot 1 \\

&= 0 + \frac{\partial}{\partial \eta} a(\eta) \\

&= \frac{\partial}{\partial \eta} a(\eta)

\end{align*}
$$
The mean of an exponential family distribution is the first derivative of the log-partition function w.r.t. the natural parameter.

(b)
Proof:
$$
\begin{align*}

\frac{\partial^2}{\partial \eta^2} p(y; \ \eta) &= \frac{\partial}{\partial \eta} \big( y \ p(y; \ \eta) - p(y; \ \eta) \frac{\partial}{\partial \eta} a(\eta) \big) \\

&= p(y; \ \eta) + y^2 p(y; \ \eta) - 2 \ y \ p(y; \ \eta) \frac{\partial}{\partial \eta} a(\eta) + p(y; \ \eta) \big( \frac{\partial}{\partial \eta} a(\eta) \big)^2 - p(y; \ \eta) \frac{\partial^2}{\partial \eta^2} a(\eta) \\

&= p(y; \ \eta) - p(y; \ \eta) \frac{\partial^2}{\partial \eta^2} a(\eta) + \big( y - \frac{\partial}{\partial \eta} a(\eta) \big)^2 p(y; \ \eta)

\end{align*}
$$
which implies:
$$\big( y - \frac{\partial}{\partial \eta} a(\eta) \big)^2 p(y; \ \eta) = \frac{\partial^2}{\partial \eta^2} p(y; \ \eta) - p(y; \ \eta) + p(y; \ \eta) \frac{\partial^2}{\partial \eta^2} a(\eta)$$
Hence,
$$
\begin{align*}

\mathrm{Var} [ \ Y; \ \eta \ ] & = \int (y - \mathbb{E} [ \ Y; \ \eta \ ])^2 p(y; \ \eta) \ dy \\

&= \int \big( y - \frac{\partial}{\partial \eta} a(\eta) \big)^2 p(y; \ \eta) \ dy \\

&= \int \frac{\partial^2}{\partial \eta^2} p(y; \ \eta) - p(y; \ \eta) + p(y; \ \eta) \frac{\partial^2}{\partial \eta^2} a(\eta) \ dy \\

&= \int \frac{\partial^2}{\partial \eta^2} p(y; \ \eta) \ dy - \int p(y; \ \eta) \ dy + \int p(y; \ \eta) \frac{\partial^2}{\partial \eta^2} a(\eta) \ dy \\

&= \frac{\partial^2}{\partial \eta^2} \int p(y; \ \eta) \ dy - \int p(y; \ \eta) \ dy + \frac{\partial^2}{\partial \eta^2} a(\eta) \int p(y; \ \eta) \ dy \\

&= 1 -1 + \frac{\partial^2}{\partial \eta^2} a(\eta) \\

&= \frac{\partial^2}{\partial \eta^2} a(\eta)

\end{align*}
$$
The variance of an exponential family distribution is the second derivative of the log-partition function w.r.t. the natural parameter.

(c)
Recall the NLL
$$
\begin{align*}

\ell (\theta) & = - \log p(y^{(i)}; \ \eta) \\

&= - \log b(y^{(i)}) \exp (\eta^T T(y^{(i)}) - a(\eta)) \\

&= a(\eta) - \eta^T y^{(i)} - \log b(y^{(i)}) \\

&= a(\theta^T x) - x^T \theta y - \log b(y)

\end{align*}
$$
We can easily compute the gradient of $\ell$
$$\nabla_\theta \ell (\theta) = x \frac{\partial}{\partial \theta} a(\theta^T x) - yx$$
So the Hessian of $\ell$ is$$H = \nabla_\theta^2 \ell (\theta) = x x^T \frac{\partial^2}{\partial \theta^2} a(\theta^T x)$$
For any $z \in \mathbb{R}^n$, we have
$$
\begin{align*}

z^T H z & = z^T \big( x x^T \frac{\partial^2}{\partial \theta^2} a(\theta^T x) \big) z \\

&= z^T x x^T z \frac{\partial^2}{\partial \theta^2} a(\theta^T x) \\

&= (x^T z)^2 \frac{\partial^2}{\partial \theta^2} a(\eta) \\

&= (x^T z)^2 \ \mathrm{Var} [ \ Y; \ \eta \ ] \\

& \geq 0

\end{align*}
$$
which means the Hessian is PSD, showing that the NLL loss of GLM is convex.

### 5. Locally weighted linear regression

(a)
i
Recall the cost function
$$
\begin{align*}

J(\theta) & = \frac{1}{2} \sum_{i = 1}^{m} w^{(i)} \big( \theta^T x^{(i)} - y^{(i)} \big)^2 \\

& = (X \theta - y)^T W (X \theta - y)

\end{align*}
$$
Notice that there are only $m$ $w^{(i)}$'s, and $w^{(i)}$ only takes effect on the $i$-th entry of $(X \theta - y)^T$ and $(X \theta - y)$.
Hence, $W$ is a diagonal matrix, and the above equation holds by picking:
$$
\begin{equation*}

X = \begin{bmatrix}

- (x^{(1)})^T - \\

\vdots \\

- (x^{(m)})^T - \\

\end{bmatrix}

\in \mathbb{R}^{m \times n}

\qquad

y = \begin{bmatrix}

y^{(1)} \\

\vdots \\

y^{(m)} \\

\end{bmatrix}

\qquad

W = \frac{1}{2} \mathrm{diag} (w^{(i)}, \dots , w^{(m)})

\end{equation*}
$$

ii
To simplify the calculation, let $J(\theta) = \frac{1}{2} (X \theta - y)^T W (X \theta - y)$. To minimize $J$, take derivative w.r.t $\theta$ and set to 0:
$$
\begin{align*}

\nabla_\theta J(\theta) &= \nabla_\theta \big( \frac{1}{2} (X \theta - y)^T W (X \theta - y) \big) \tag{1.5.1} \\

&= \nabla_\theta \big((X^{T}\theta^{T} - y^{T})W(x\theta - y) \big) \tag{1.5.2} \\

&= \nabla_\theta (X^{T}\theta^{T}W - y^{T}W)(X\theta - y) \tag{1.5.3} \\

&= \nabla_\theta (X^{T}\theta^{T}WX\theta - X^{T}\theta^{T}Wy - y^{T}WX\theta - y^{T}Wy) \tag{1.5.4} \\

&= \nabla_\theta (X^{T}\theta^{T}WX\theta - 2X^{T}\theta^{T}Wy - y^{T}Wy) \tag{1.5.5} \\

&= \nabla_\theta (X^{T}\theta^{T}WX\theta - 2(X^{T}\theta^{T}Wy)^{T}) \tag{1.5.6} \\

&= \nabla_\theta (X^{T}\theta^{T}WX\theta - 2X\theta^{T}y^{T}) \tag{1.5.7} \\

&= \nabla_\theta (X^{T}\theta^{T}WX\theta - 2y^{T}WX\theta \tag{1.5.8} \\

&= 2X^{T}WX\theta - 2X^{T}Wy \tag{1.5.9}

\end{align*}
$$
$W^{T} = W$ Symmetric Matrix
(1.5.4)
$$
\begin{align*}

y^{T}WX\theta &= X\theta Wy^{T} \\

&= (X\theta Wy^{T})^{T} \\

&= X^{T}\theta^{T}W^{T}y \\

&= X^{T}\theta^{T}Wy

\end{align*}
$$

(1.5.8)
$\nabla_\theta J(\theta)= 0$
$$
\begin{align*}

\nabla_\theta (2y^{T}WX\theta) &= 2(y^{T}WX)^{T} \\

&= 2(yW^{T}X^{T}) \\

&= 2X^{T}Wy

\end{align*}
$$
By solving the equation, we obtain:

$$\theta = (X^T W X)^{-1} X^T W y$$

iii
$$
\begin{align*}

\ell (\theta) &= \sum_{i = 1}^{m} \log \frac{1}{\sqrt{2 \pi} \sigma^{(i)}} \exp \big( - \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2 (\sigma^{(i)})^2} \big) \\

&= - m \log \sqrt{2 \pi} \sigma^{(i)} - \sum_{i = 1}^{m} \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2 (\sigma^{(i)})^2} \\

&= - \frac{1}{2} \sum_{i = 1}^{m} \frac{1}{(\sigma^{(i)})^2} (\theta^T x^{(i)} - y^{(i)})^2 - m \log \sqrt{2 \pi} \sigma^{(i)}

\end{align*}
$$
Thus, maximizing $\ell (\theta)$ is equivalent to minimizing
$$\frac{1}{2} \sum_{i = 1}^{m} \frac{1}{(\sigma^{(i)})^2} (\theta^T x^{(i)} - y^{(i)})^2$$
By setting $w^{(i)} = 1 / (\sigma^{(i)})^2$, finding the maximum likelihood estimate of $\theta$ reduces to minimizing $J(\theta)$.