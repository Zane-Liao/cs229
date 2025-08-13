# PS3-Solution

### 1. Simple Nerual Network

##### (a)

$$
\begin{align*}

\text{We Know} \\
z^{[1]} = w^{[1]}_{i,j}x + w^{[1]}_{0,j} \\
h_{j} = \sigma(z^{[1]}) \\
z^{[2]} = w^{[2]}_{j}h_{j} + w^{[2]}_{0} \\
o = \sigma(z^{[2]}) \\
l=\frac{1}{m}\sum_{i=1}^{m}(o^{(i)}-y^{(i)})^{2} \\

\frac{\partial l}{\partial w^{[1]}_{1, 2}} &= \frac{\partial l}{\partial o}\frac{\partial o}{\partial z^{[2]}} \frac{\partial z^{[2]}}{\partial h_{2}}
\frac{\partial h_{2}}{\partial z^{[1]}} \frac{\partial z^{[1]}}{\partial w^{[1]}_{1,2}} \\
&= \frac{1}{m}\sum_{i=1}^{m}2(o^{(i)}-y^{(i)})\cdot o^{(i)}(1-o^{(i)})\cdot w^{[2]}_2\cdot h_2^{(i)}(1-h_2^{(i)})\cdot x_1^{(i)} \\

h^{(i)}_2 = w^{[1]}_{1,2}x^{(i)}_1 + w^{[1]}_{2,2}x^{(i)}_2 + w^{[1]}_{0,2} \\

w^{[1]}_{1,2} &:= w^{[1]}_{1,2} - \alpha\cdot \frac{\partial l}{\partial w^{[1]}_{1,2}} \\
 &:= w^{[1]}_{1,2} - \alpha\cdot \frac{1}{m}\sum_{i=1}^{m}2(o^{(i)}-y^{(i)})\cdot o^{(i)}(1-o^{(i)})\cdot w^{[2]}_2\cdot h_2^{(i)}(1-h_2^{(i)})\cdot x_1^{(i)} \\
 &:= w^{[1]}_{1,2} - \alpha\cdot \frac{2}{m}\sum_{i=1}^{m}w^{[2]}_2\cdot (o^{(i)}-y^{(i)})\cdot o^{(i)}(1-o^{(i)})\cdot h_2^{(i)}(1-h_2^{(i)})\cdot x_1^{(i)} \\

\end{align*}
$$

##### (b)

- When assuming that the positive class is x1<= 0.5, x2 <= 0.5, x1 + x2=>4, and the rest are negative classes, there is a set of weights that makes the classification accuracy 100%(This question is very confusing, something is not quite right, I think some necessary conditions are missing, please refer to the Solution or skip it directly)

##### (c)
Solution
- It's not possible. When we adopt linear function for the hidden layer and step function for the
output, the entire neuron network can be viewed as one linear classifier (not three). Because the
dataset is not linearly separable, so it's impossible to achieve 100% accuracy.

### 2. KL divergence and Maximum Likelihood

##### (a)
Prove: $\forall P, Q \quad D_{KL}(P || Q) \geq 0$<br>
$D_{KL}(P || Q) = 0 \text{ if and only if } P = Q$

$$
\begin{align*}

D_{KL}(P||Q) &= \sum_{x \in \mathcal{X}} P(x) \log\frac{P(x)}{Q(x)} \\
&= - \sum_{x \in \mathcal{X}} P(x) \log\frac{Q(x)}{P(x)} \\
\big (\text{Use Jensen' inequality: $E[f(X)] \geq f(E[X])$} \big)
&= -E[-\log\frac{Q(x)}{P(x)}] \geq -\log E[\frac{Q(x)}{P(x)}] \\
\text{So, We get}
&= -log\left(\sum_{x \in \mathcal{X}} P(x) \frac{Q(x)}{P(x)} \right) \\
&= -log\sum_{x \in \mathcal{X}} Q(x) \\
\big(\text{X = E[X] probability is 1}\big)
&= -log1 \\
&= 0

\end{align*}
$$

-  If $P = Q$, then $D_{KL}(P||Q) = \sum_{x \in \mathcal{X}} P(x)\log 1 = 0$.
-  If $D_{KL}(P||Q) = 0$, then $\frac{P(x)}{Q(x)} =  E[\frac{P(x)}{Q(x)}] = 1$, $P = Q$.
-  So we get $D_{KL}(P || Q) = 0 \text{ if and only if } P = Q$

##### (b)
Prove:
$D_{KL}(P(X,Y) || Q(X,Y)) = D_{KL}(P(X) || Q(X)) + D_{KL}(P(Y|X) || Q(Y|X))$

$$
\begin{align*}

D_{KL}(P(X,Y) || Q(X,Y)) &= \sum_x \sum_y P(x,y) \log \frac{P(x,y)}{Q(x,y)} \\
&= \sum_x \sum_y P(x)P(y|x) \log \frac{P(x)P(y|x)}{Q(x)Q(y|x)} \\
&= \sum_x \sum_y P(x)P(y|x) \left( \log \frac{P(x)}{Q(x)}+\log\frac{P(y|x)}{Q(y|x)} \right) \
\\
&= \sum_x \sum_y P(x)P(y|x) \log \frac{P(x)}{Q(x)} + \sum_x \sum_y P(x)P(y|x) \log\frac{P(y|x)}{Q(y|x)} \\
&= \sum_x P(x) \log \frac{P(x)}{Q(x)} \sum_y P(y|x) + \sum_x P(x) \sum_y P(y|x) \log\frac{P(y|x)}{Q(y|x)} \\
&= \sum_x P(x) \log \frac{P(x)}{Q(x)} + \sum_y P(y) \left( \sum_x P(x|y) \log \frac{P(x|y)}{Q(x|y)} \right) \\
&= D_{KL}(P(X) || Q(X)) + D_{KL}(P(Y|X) || Q(Y|X)) \quad \text{to End!!!} \\

\end{align*}
$$

##### (c)
Prove: $\arg \min_\theta D_{KL}(\hat{P} | P_\theta) = \arg \max_\theta \sum_{i=1}^m \log P_\theta(x^{(i)})$

$$
\begin{align*}

D_{KL}(\hat{P}||P_{\theta}) &= \sum_{x \in \mathcal{X}}\hat{P}(x) \log \frac{{\hat{P}(x)}}{P_{\theta}(x)} \\
&= \sum_{x \in \mathcal{X}}\hat{P}(x) \log {\hat{P}(x)} - 
\sum_{x \in \mathcal{X}}\hat{P}(x) \log {P_{\theta}(x)} \\

\arg \min_\theta D_{KL}(\hat{P} | P_\theta) &= \arg \min_\theta \sum_{x \in \mathcal{X}}\hat{P}(x) \log {\hat{P}(x)} - \sum_{x \in \mathcal{X}}\hat{P}(x) \log {P_{\theta}(x)} \\
&= \arg \max_\theta \sum_{x \in \mathcal{X}}\hat{P}(x) \log {P_{\theta}(x)} \\
&=  \arg \max_\theta \sum_{x \in \mathcal{X}}\hat{P}(x) \left(\frac{1}{m} \sum_{i=1}^m \mathbf{1}{\{x^{(i)} = x\}}\right) \log {P_{\theta}(x)} \\
&= \arg \max_\theta \sum_{i=1}^m \log P_\theta(x^{(i)})

\end{align*}
$$

### 3. KL Divergence, Fisher Information, and the Natural Gradient

##### (a)
Prove: $E_{y \sim p(y;\theta)}[\nabla_{\theta'} \log p(y; \theta')|_{\theta' = \theta}] = 0$

$$
\begin{align*}
\nabla_{\theta} \log p(y;\theta) = \frac{\nabla_{\theta}p(y;\theta)}{p(y;\theta)}  \\
\text{We Know $E_{y \sim p(y)}[g(y)] = \int_{-\infty}^{\infty} p(y)g(y)dy$ } \\
E_{y \sim p(y;\theta)}[\nabla_{\theta'} \log p(y; \theta')|_{\theta' = \theta}] &= E_{y \sim p(y;\theta)} \left[ \frac{\nabla_{\theta}p(y;\theta)}{p(y;\theta)} \right] \\
&= \int^{\infty}_{-\infty} p(y; \theta) \frac{\nabla_{\theta}p(y;\theta)}{p(y;\theta)} dy \\
&= \int^{\infty}_{-\infty} \nabla_{\theta}p(y;\theta) dy \\
&= \nabla_{\theta} \int^{\infty}_{-\infty} p(y;\theta) dy \\
\text{We Know Probability density normalization $\int_{-\infty}^\infty p(y; \theta) \, dy = 1$ } \\
&= \nabla_{\theta} 1 \\
&= 0

\end{align*}
$$

##### (b)
Prove: $I(\theta) = E_{y \sim p(y;\theta)}[\nabla_{\theta'} \log p(y; \theta') \nabla_{\theta'} \log p(y; \theta')^T |_{\theta' = \theta}]$

$$
\begin{align*}
Cov[X] = E[(X-E[X])(X-E[X])^{T}] = E[XX]^{T} \\
\\
\text{E[X] = 0, We Know $I(\theta) = \text{Cov}_{y \sim p(y;\theta)}[\nabla_{\theta'} \log p(y; \theta')|_{\theta' = \theta}]$} \\
\\
I(\theta) = \text{Cov}_{y \sim p(y;\theta)}[\nabla_{\theta'} \log p(y; \theta')|_{\theta' = \theta}] = E_{y \sim p(y;\theta)}[\nabla_{\theta'} \log p(y; \theta') \nabla_{\theta'} \log p(y; \theta')^T |_{\theta' = \theta}] \\

\end{align*}
$$

##### (c)
Prove: $E_{y \sim p(y;\theta)}[-\nabla^2_{\theta'} \log p(y; \theta')|_{\theta' = \theta}] = I(\theta)$

$$
\begin{align*}


\end{align*}
$$

##### (d)
Prove: $D_{KL}(p_\theta || p_{\theta+d}) \approx \frac{1}{2} d^T I(\theta) d$

$$
\begin{align*}


\end{align*}
$$

##### (e)
Problem: Natural Gradient

$$
\begin{align*}


\end{align*}
$$

##### (f)
Problem: Relationship between Newton's method and natural gradient

$$
\begin{align*}


\end{align*}
$$

### 4. Semi-Supervised Learning

##### (a)
Prove: $\ell_{\text{semi-sup}}(\theta^{(t+1)}) \geq \ell_{\text{semi-sup}}(\theta^{(t)})$<br>

We know that proving the above is to prove that the following function increases monotonically in each iteration
$$
\begin{align*}

\ell_{\text{semi-sup}}(\theta^{(t+1)}) = \ell_{\text{unsup}}(\theta^{(t+1)}) + \alpha\ell_{\text{sup}}(\theta^{(t+1)}) \\
\\
\text{For the unsupervised part, Constructing the lower buond of Jensen's inequality(E-Step)} \\
\\
\ell_{\text{unsup}}(\theta^{(t+1)})
= \sum_{i=1}^m \log p\bigl(x^{(i)};\,\theta^{(t+1)}\bigr)
\;\ge\;
\sum_{i=1}^m \left(\sum_{z^{(i)}} Q_i^{(t)}(z^{(i)}) 
\log \frac{p\bigl(x^{(i)},z^{(i)};\,\theta^{(t+1)}\bigr)}
{Q_i^{(t)}(z^{(i)})}\right) \\
\\
\log \sum_z a_z \;\ge\; \sum_z q_z\,\log\frac{a_z}{q_z} \\
\\
\text{The new lower bound is at least as large as the old one(M-Step)} \\
\\
\sum_i\sum_z Q_i^{(t)}(z)\log\frac{p(x^{(i)},z;\theta^{(t+1)})}{Q_i^{(t)}(z)}
\;\ge\;
\sum_i\sum_z Q_i^{(t)}(z)\log\frac{p(x^{(i)},z;\theta^{(t)})}{Q_i^{(t)}(z)}
\\
\\
\text{For the supervised part, We Know log-likelihood } \\
\\
\ell_{\text{sup}}(\theta)
=\sum_{i=1}^{\tilde m}\log p\bigl(\tilde x^{(i)},\tilde z^{(i)};\theta\bigr) \\
\\
\text{Add it directly in the M-Step to maximize and ensure} \\
\\
\ell_{\text{sup}}(\theta^{(t+1)})
=\sum_{i=1}^{\tilde m}\log p\bigl(\tilde x^{(i)},\tilde z^{(i)};\theta^{(t+1)}\bigr)
\;\ge\;
\ell_{\text{sup}}(\theta)
=\sum_{i=1}^{\tilde m}\log p\bigl(\tilde x^{(i)},\tilde z^{(i)};\theta\bigr) \\
\\
\text{Last, We Merge => Strictly increasting} \\
\\
\ell_{\text{semi-sup}}(\theta^{(t+1)})
\;\ge\;
\ell_{\text{semi-sup}}(\theta^{(t)})

\end{align*}
$$

##### (b)

Semi-supervised E-Step for GMM(Gaussian mixture model)<br>
Estimating latent variables $z^{(i)}$

$$
\begin{align*}
\text{Posterior probability} \\
w^{(i)}_j &= Q_i(z^{(i)} = j) = P(z^{(i)}_j = j|x^{(i)};\phi,\mu,\Sigma) \\
&= \frac{P(x^{(i)} | z^{(i)} = j; \mu, \Sigma) P(z^{(i)} = j ; \phi)}{\sum^{k}_{l=1} P(x^{(i)} | z^{(i)} = l; \mu, \Sigma) P(z^{(i)} = l; \phi)} \\
&= \frac{\frac{1}{(2\pi)^{n/2}|\Sigma_j|^{1/2}} \exp\left(-\frac{1}{2}(x^{(i)} - \mu_j)^T \Sigma^{-1}_j(x^{(i)} - \mu_j)\right)\phi_j}{\sum^{k}_{l=1}\frac{1}{(2\pi)^{n/2}|\Sigma_l|^{1/2}} \exp\left(-\frac{1}{2}(x^{(i)} - \mu_l)^T \Sigma^{-1}_j(x^{(i)} - \mu_l)\right)\phi_l}

\end{align*}
$$

##### (C)

Semi-supervised M-Step for GMM(Gaussian mixture model)<br>
Write out to total log-likelihood $\ell_{\text{semi-sup}} = \ell_{\text{unsup}} + \alpha\ell_{\text{sup}}$<br>
We need to reduce $\mu, \Sigma, \phi$ Closed-form expression


###### (1) Unlabeled Data - Log-likelihood in GMM

$$\ell_{\rm obs}(\theta) = \sum_{i=1}^m \log p(x^{(i)};\theta) = \sum_{i=1}^m \log\left(\sum_{j=1}^k \phi_j\,\mathcal{N}(x^{(i)}\mid \mu_j,\Sigma_j)\right)$$

**Complete data log-likelihood**

$$\ell_{\rm comp}(\theta) = \sum_{i=1}^m \log p\left(x^{(i)},z^{(i)};\theta\right) = \sum_{i=1}^m \sum_{j=1}^k \mathbf{1}\{z^{(i)}=j\}\,\left[\log \phi_j + \log \mathcal{N}(x^{(i)}\mid\mu_j,\Sigma_j)\right]$$

**We use**

$$w_j^{(i)} = p\left(z^{(i)}=j\,\bigg|\,x^{(i)};\theta^{(t)}\right)$$

$w_j^{(i)}$ replaces $\mathbf{1}\{z^{(i)}=j\}$

$$\boxed{\ell_{\rm unsup} = \sum_{i=1}^m \sum_{j=1}^k w_j^{(i)}\left[\log \phi_j + \log \mathcal{N}(x^{(i)}\mid \mu_j,\Sigma_j)\right]}$$

**Where**

$$\log \mathcal{N}(x\mid \mu_j,\Sigma_j) = -\frac{1}{2} (x-\mu_j)^\top \Sigma_j^{-1}(x-\mu_j)$$

$$\ell_{\rm unsup}(\mu_j) = \sum_{i=1}^m w^{(i)}_j \left[-\frac{1}{2} (x^{(i)}-\mu_j)^\top \Sigma_j^{-1}(x^{(i)}-\mu_j)\right] + \text{const}$$

$$\frac{\partial}{\partial \mu}\left[-\frac{1}{2} (x-\mu)^\top \Sigma^{-1}(x-\mu)\right] = \Sigma^{-1}(x-\mu)$$

$$\nabla_{\mu_j}{\ell_{\rm unsup}} = \sum_{i=1}^{m}w^{(i)}_j\Sigma^{-1}_j(x^{(i)} - \mu_j)$$

###### (2) Adding Labels to Data

$$\ell_{\rm sup}(\theta) = \sum_{i=1}^{\tilde m} \log p\left(\tilde x^{(i)},\,\tilde z^{(i)};\theta\right)$$

**GMM Generative Process**

$$p(\tilde x,\tilde z=j) = \phi_j\;\mathcal{N}(\tilde x\mid\mu_j,\Sigma_j)$$

**We get**

$$\boxed{\ell_{\rm sup} = \sum_{i=1}^{\tilde m} \log\left[\phi_{\tilde z^{(i)}}\, \mathcal{N}(\tilde x^{(i)}\mid \mu_{\tilde z^{(i)}},\Sigma_{\tilde z^{(i)}})\right]}$$

This is useful only when the label = j, so we use $\mathbf{1}\{\tilde z^{(i)}=j\}$

**We get**

$$\ell_{\rm sup}(\mu_j) = \sum_{i=1}^{\tilde m} \mathbf{1}\{\tilde z^{(i)}=j\} \left[-\frac{1}{2}(\tilde x^{(i)}-\mu_j)^\top\Sigma_j^{-1}(\tilde x^{(i)}-\mu_j)\right] + \text{const}$$

$$\nabla_{\mu_j}\,\ell_{\rm sup} = \sum_{i=1}^{\tilde m} \mathbf{1}\{\tilde z^{(i)}=j\}\,\Sigma_j^{-1}\left(\tilde x^{(i)}-\mu_j\right)$$

###### (3) Derivation

$$\nabla_{\mu_j}{\ell_{\rm unsup}} = \sum_{i=1}^{m}w^{(i)}_j\Sigma^{-1}_j(x^{(i)} - \mu_j) = \Sigma^{-1}_j\left(\sum_{i=1}^{m}w^{(i)}_jx^{(i)} - \sum_{i=1}^{m}w^{(i)}_j\mu_j\right)$$

$$\nabla_{\mu_j}{\ell_{\rm sup}} = \sum_{i=1}^{\tilde{m}}\mathbf{1}\{\tilde z^{(i)} = j\}\Sigma^{-1}_j(\tilde x^{(i)} - \mu_j) = \Sigma^{-1}_j\left(\sum_{i=1}^{\tilde{m}}\mathbf{1}\{\tilde z^{(i)} = j\}\tilde x^{(i)} - \sum_{i=1}^{\tilde{m}}\mathbf{1}\{\tilde z^{(i)} = j\}\mu_j\right)$$

**Merge**

$$\nabla_{\mu_j}\ell_{\text{semi-sup}} = \nabla_{\mu_j}\ell_{\text{unsup}} + \alpha \nabla_{\mu_j}\ell_{\text{sup}}$$

$$= \Sigma^{-1}_j\left[\left(\sum_{i=1}^{m}w^{(i)}_jx^{(i)} - \sum_{i=1}^{m}w^{(i)}_j\mu_j\right) + \alpha \left(\sum_{i=1}^{\tilde{m}}\mathbf{1}\{\tilde z^{(i)} = j\}\tilde x^{(i)} - \sum_{i=1}^{\tilde{m}}\mathbf{1}\{\tilde z^{(i)} = j\}\mu_j\right)\right]$$

$$= \Sigma^{-1}_j\left[\left(\sum_{i=1}^{m}w^{(i)}_jx^{(i)} + \alpha\sum_{i=1}^{\tilde{m}}\mathbf{1}\{\tilde z^{(i)} = j\}\tilde x^{(i)}\right) - \mu_j\left(\sum_{i=1}^{m}w^{(i)}_j + \alpha\sum_{i=1}^{\tilde{m}}\mathbf{1}\{\tilde z^{(i)} = j\}\right)\right] = 0$$

**For $\mu_j$**

$$\mu_j = \frac{\sum_{i=1}^{m}w^{(i)}_jx^{(i)} + \alpha\sum_{i=1}^{\tilde{m}}\mathbf{1}\{\tilde z^{(i)} = j\}\tilde x^{(i)}}{\sum_{i=1}^{m}w^{(i)}_j + \alpha\sum_{i=1}^{\tilde{m}}\mathbf{1}\{\tilde z^{(i)} = j\}}$$

### Derivation for $\Sigma_j$

###### (1) Complete-data log-likelihood

**Joint Probability**

$$p(x^{(i)}, z^{(i)}=j \mid \theta) = \phi_j\;\mathcal{N}\left(x^{(i)}\mid\mu_j,\Sigma_j\right)$$

**Complete-data log-likelihood**

$$\ell_{\rm comp}(\theta) = \sum_{i=1}^m \log p\left(x^{(i)},z^{(i)};\theta\right) = \sum_{i=1}^m \sum_{j=1}^k \mathbf{1}\{z^{(i)}=j\}\,\left[\log\phi_j + \log\mathcal{N}(x^{(i)}\mid\mu_j,\Sigma_j)\right]$$

**Compute responsibilities**

$$w_j^{(i)} = p\left(z^{(i)}=j \mid x^{(i)};\theta^{(t)}\right) = \mathbb{E}\left[\mathbf{1}\{z^{(i)}=j\}\mid x^{(i)}\right]$$

$$\boxed{\ell_{\rm unsup} = \sum_{i=1}^m \sum_{j=1}^k w_j^{(i)}\, \log\mathcal{N}\left(x^{(i)}\mid\mu_j,\Sigma_j\right)}$$

###### (2) Adding labels to data

$$\sum_{i=1}^{\tilde m}\log p\left(\tilde x^{(i)},\tilde z^{(i)};\theta\right) = \sum_{i=1}^{\tilde m}\sum_{j=1}^k \mathbf{1}\{\tilde z^{(i)}=j\} \left[\log\phi_j + \log\mathcal{N}(\tilde x^{(i)}\mid\mu_j,\Sigma_j)\right]$$

$$\boxed{\ell_{\rm sup} = \sum_{i=1}^{\tilde m} \mathbf{1}\{\tilde z^{(i)}=j\}\, \log\mathcal{N}\left(\tilde x^{(i)}\mid\mu_j,\Sigma_j\right)}$$

###### (3) Derivation

$$\log \mathcal{N}(x\mid\mu_j,\Sigma_j) = -\frac{1}{2}\log\left|(2\pi)\Sigma_j\right| -\frac{1}{2}\,(x-\mu_j)^\top \Sigma_j^{-1}(x-\mu_j)$$

**We use**

$$f(\Sigma_j) = -\frac{1}{2}\log|\Sigma_j| -\frac{1}{2}\,(x-\mu_j)^\top \Sigma_j^{-1}(x-\mu_j)$$

$$\nabla_{\Sigma_j}\left[-\frac{1}{2}\log|\Sigma_j|\right] = -\frac{1}{2}\,\Sigma_j^{-1}$$

$$\nabla_{\Sigma_j}\left[-\frac{1}{2}(x-\mu_j)^\top \Sigma_j^{-1}(x-\mu_j)\right] = -\frac{1}{2} \left[-\,\Sigma_j^{-1}(x-\mu_j)(x-\mu_j)^\top\Sigma_j^{-1}\right]$$

$$= \frac{1}{2}\,\Sigma_j^{-1}(x-\mu_j)(x-\mu_j)^\top\Sigma_j^{-1}$$

**Combining**

$$\nabla_{\Sigma_j}\log\mathcal{N}(x\mid\mu_j,\Sigma_j) = \frac{1}{2}\left[\Sigma_j^{-1}(x-\mu_j)(x-\mu_j)^\top\Sigma_j^{-1} - \Sigma_j^{-1}\right]$$

$$\nabla_{\Sigma_j}{\ell_{\rm unsup}} = \sum_{i=1}^m w^{(i)}_j\;\frac{1}{2}\left[ \Sigma_j^{-1}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^\top\Sigma_j^{-1} -\Sigma_j^{-1}\right]$$

$$\nabla_{\Sigma_j}{\ell_{\rm sup}} = \sum_{i=1}^{\tilde m}\mathbf{1}\{\tilde z^{(i)}=j\}\;\frac{1}{2}\left[ \Sigma_j^{-1}(\tilde x^{(i)}-\mu_j)(\tilde x^{(i)}-\mu_j)^\top\Sigma_j^{-1} -\Sigma_j^{-1}\right]$$


$\nabla_{\Sigma_j}\ell_{\text{semi-sup}} = \nabla_{\Sigma_j}\ell_{\text{unsup}} + \alpha \nabla_{\Sigma_j}\ell_{\text{sup}}$

$= -\frac{1}{2}\sum_{i=1}^{m} w_j^{(i)} \Sigma_j^{-1} + \frac{1}{2}\Sigma_j^{-1} \left(\sum_{i=1}^{m} w_j^{(i)} (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T\right) \Sigma_j^{-1}$

$-\frac{1}{2}\alpha \sum_{i=1}^{\tilde{m}} \mathbf{1}{\tilde{z}^{(i)} = j}\Sigma_j^{-1} + \frac{1}{2}\alpha\Sigma_j^{-1} \left(\sum_{i=1}^{\tilde{m}} \mathbf{1}{\tilde{z}^{(i)} = j} (\tilde{x}^{(i)} - \mu_j)(\tilde{x}^{(i)} - \mu_j)^T\right) \Sigma_j^{-1}$

$= -\frac{1}{2}\Sigma_j^{-1} \left(\sum_{i=1}^{m} w_j^{(i)} + \alpha \sum_{i=1}^{\tilde{m}} \mathbf{1}{\tilde{z}^{(i)} = j}\right)$

$+ \frac{1}{2}\Sigma_j^{-1} \left(\sum_{i=1}^{m} w_j^{(i)} (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T + \alpha \sum_{i=1}^{\tilde{m}} \mathbf{1}{\tilde{z}^{(i)} = j} (\tilde{x}^{(i)} - \mu_j)(\tilde{x}^{(i)} - \mu_j)^T\right) \Sigma_j^{-1}$
$= 0$

**For $\Sigma_j$**

$\Sigma_j = \frac{\sum_{i=1}^{m} w_j^{(i)} (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T + \alpha \sum_{i=1}^{\tilde{m}} \mathbf{1}{\tilde{z}^{(i)} = j} (\tilde{x}^{(i)} - \mu_j)(\tilde{x}^{(i)} - \mu_j)^T}{\sum_{i=1}^{m} w_j^{(i)} + \alpha \sum_{i=1}^{\tilde{m}} \mathbf{1}{\tilde{z}^{(i)} = j}}$

#### Derivation for $\phi_j$

$$
\begin{align*}
% Image 1 expressions
\mathcal{L}(\phi) = \sum_{i=1}^{m} \sum_{l=1}^{k} w_l^{(i)} \log \phi_l + \sum_{i=1}^{\tilde{m}} \sum_{l=1}^{k} \mathbf{1}\{z^{(i)} = l\} \log \phi_l + \beta\left(\sum_{l=1}^{k} \phi_l - 1\right)
\\
\nabla_{\phi_j} \mathcal{L}(\phi) = \sum_{i=1}^{m} \frac{w_j^{(i)}}{\phi_j} + \sum_{i=1}^{\tilde{m}} \frac{\mathbf{1}\{z^{(i)} = j\}}{\phi_j} + \beta = 0
\\
\phi_j = \frac{\sum_{i=1}^{m} w_j^{(i)} + \alpha \sum_{i=1}^{\tilde{m}} \mathbf{1}\{z^{(i)} = j\}}{-\beta}
\\
% Image 2 expressions
\sum_{l=1}^{k} \phi_l = \frac{\sum_{i=1}^{m} \sum_{l=1}^{k} w_l^{(i)} + \alpha \sum_{i=1}^{\tilde{m}} \sum_{l=1}^{k} \mathbf{1}\{z^{(i)} = l\}}{-\beta}
\\
= \frac{m + \alpha\tilde{m}}{-\beta}

= 1
\\
-\beta = m + \alpha\tilde{m}
\\
\phi_j = \frac{\sum_{i=1}^{m} w_j^{(i)} + \alpha \sum_{i=1}^{\tilde{m}} \mathbf{1}\{z^{(i)} = j\}}{m + \alpha\tilde{m}} \\
\end{align*}
$$

