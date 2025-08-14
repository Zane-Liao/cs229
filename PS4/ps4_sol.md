# PS4-Solution

## 1. Convolutional Neural Networks: MNIST image classification(code Problem...)

## 2. Oﬀ Policy Evaluation And Causal Inference

### (a) Importance Sampling. Prove: if $\hat{\pi}_0 = \pi_0$, $E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} R(s, a)$ = $E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)$

$$
\begin{align*}
\text{We Know, } \\
E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)
&= \sum_{(s,a)} R(s, a)p(s, a) \\
&= \sum_{(s,a)} R(s, a)p(s)p(a|s) \\ 
&= \sum_{(s,a)} R(s, a)p(s)\pi_1(s, a) \\

\text{So, } \\
E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{\hat\pi_0(s,a)}R(s,a) 
&= E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{\pi_0(s,a)}R(s,a) \\
&= \frac{\pi_1(s,a)}{\pi_0(s,a)} \sum_{(s,a)} R(s, a)p(s)\pi_0(s, a) \\
&= \sum_{(s,a)} R(s, a)p(s)\pi_1(s,a) \\
&= E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)

\end{align*}
$$

### (b) Weighted Importance Sampling. Prove: if $\hat{\pi}_0 = \pi_0$, $\frac{E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{\hat{\pi}_0(s,a)} R(s, a)}{E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{\hat{\pi}_0(s,a)}}$ = $E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)$

$$
\begin{align*}
\text{Same Reason, We Get:} \\
\frac{E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{\hat{\pi}_0(s,a)} R(s, a)}{E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{\hat{\pi}_0(s,a)}} &= \frac{E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{{\pi}_0(s,a)} R(s, a)}{E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{{\pi}_0(s,a)}} \\
&= \frac{\frac{\pi_1(s,a)}{\pi_0(s,a)} \sum_{(s,a)} R(s, a)p(s)\pi_0(s, a)} {\frac{\pi_1(s,a)}{\pi_0(s,a)} \sum_{(s,a)} p(s)\pi_0(s, a)} \\
&= \frac{\sum_{(s,a)} R(s, a)p(s) \pi_1(s,a)} {\sum_{(s,a)} p(s) \pi_1(s,a)} \\
&= E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a) \\

\end{align*}
$$

### (c) Please show that the weighted importance sampling estimator is biased in these situations

$$
\begin{align*}

\frac{E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{\hat{\pi}_0(s,a)} R(s, a)}{E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{\hat{\pi}_0(s,a)}}
&= \frac{\frac{\pi_1(s,a)}{\hat\pi_0(s,a)} \sum_{(s,a)} p(s, a) R(s, a)} {\frac{\pi_1(s,a)}{\hat\pi_0(s,a)} \sum_{(s,a)} p(s, a)} \\
\text{If there is only one element, then: }
&= R(s, a)
\\
E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} R(s, a) \neq E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)

\end{align*}
$$

### (d) Doubly Robust

**i. Prove: if $\hat{\pi}_0 = \pi_0$, $E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} \hat{R}(s, a)) + \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} (R(s, a) - \hat{R}(s, a))\right)$ = $E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)$**

$$
\begin{aligned}

& E_{s \sim p(s), a \sim \pi_0(s,a)}\left(E_{s \sim p(s), a \sim \pi_1(s,a)} \hat R(s, a)\right) = E_{s \sim p(s), a \sim \pi_1(s,a)} \hat R(s, a)

\\
& E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} \hat{R}(s, a)) + \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} (R(s, a) - \hat{R}(s, a))\right) \\
&= E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} \hat{R}(s, a)) + \frac{\pi_1(s, a)}{{\pi}_0(s, a)} (R(s, a) - \hat{R}(s, a))\right) \\
&= E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} \hat{R}(s, a)) + (\frac{\pi_1(s, a)}{{\pi}_0(s, a)} R(s, a) - \frac{\pi_1(s, a)}{{\pi}_0(s, a)}  \hat{R}(s, a))\right) \\
&= E_{s \sim p(s), a \sim \pi_0(s,a)} \left(E_{a \sim \pi_1(s,a)} \hat{R}(s, a) + (E_{ a \sim \pi_1(s,a)} R(s, a) - E_{a \sim \pi_1(s,a)} \hat R(s, a))\right) \\
&= E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)


\end{aligned}
$$

**ii. Prove: if $\hat{R}(s, a) = R(s, a)$, $E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} \hat{R}(s, a)) + \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} (R(s, a) - \hat{R}(s, a))\right)$ = $E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)$**

$$
\begin{aligned}
& E_{s \sim p(s), a \sim \pi_0(s,a)}\left(E_{s \sim p(s), a \sim \pi_1(s,a)} \hat R(s, a)\right) = E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)

\\
& E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} \hat{R}(s, a)) + \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} (R(s, a) - \hat{R}(s, a))\right)
\\
&= E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} {R}(s, a)) + \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} (R(s, a) - {R}(s, a))\right) \\
&= E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} {R}(s, a)) + (\frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} R(s, a) - \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)}  {R}(s, a))\right) \\
&= E_{s \sim p(s), a \sim \pi_0(s,a)} \left(E_{a \sim \pi_1(s,a)} {R}(s, a) + (E_{ a \sim \pi_1(s,a)} R(s, a) - E_{a \sim \pi_1(s,a)} R(s, a))\right) \\
&= E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)


\end{aligned}
$$

### (e) where you might have a choice between the importance sampling estimator and the regression estimator. Please state whether the importance sampling estimator or the regression estimator would probably work best in each situation and explain why it would work better. In all of these situations, your states sconsist of patients, your actions arepresent the drugs to give to certain patients and your R(s,a) is the lifespan of the patient after receiving the drug

**i. Drugs are randomly assigned to patients, but the interaction between the
drug, patient and lifespan is very complicated**<br>

- Importance Sampling Estimator, $R(s,a)$ too complicated, Not applicable to regression estimators.


**ii. Drugs are assigned to patients in a very complicated manner, but the inter-action between the drug, patient and lifespan is very simple**<br>

- Regression estimators, $\hat \pi$ too complicated, Not applicable to Importance Sampling Estimator.

## 3. Principal components analysis(PCA). Show that the unit-length vector u that minimizes the mean squared error between projected points and original points corresponds to the first principal component for the data

Prove: $\arg \min_{u:u^T u=1} \sum_{i=1}^m ||x^{(i)} - f_u(x^{(i)})||_2^2$

$$
\begin{align*}

f_u(x) = \arg \min_{v \in V} ||x - v||^2 = uu^{T}x
\\
\arg \min_{u:u^T u=1} \sum_{i=1}^m ||x^{(i)} - f_u(x^{(i)})||_2^2 &= \arg \min_{u:u^T u=1} \sum_{i=1}^m ||x^{(i)} - uu^{T}x||_2^2 \\
&= \arg \min_{u:u^T u=1} \sum_{i=1}^m (x^{(i)} - u^{T}ux^{(i)})^{T}(x^{(i)} - u^{T}ux^{(i)}) \\
&= \arg \min_{u:u^T u=1} \sum_{i=1}^m x^{(i)^{T}}x^{(i)} - x^{(i)}u^{T}ux^{(i)} \\
&= \arg \max_{u:u^T u=1} \sum_{i=1}^m x^{(i)^{T}}u^{T}ux^{(i)} \\
&= \arg \max_{u:u^T u=1} u^{T}(\sum_{i=1}^mx^{(i)^{T}}x^{(i)})u \\
&= \arg \max_{u:u^T u=1} u^{T}(\sum_{i=1}^mx^{(i)}x^{(i)^{T}})u

\end{align*}
$$

## 4. Independent components analysis(ICA)

### (a) Gaussian source. Try to derive a closed form expression for W in terms of X when g is the standard normal CDF

$$
\begin{aligned}
& \text{We Know $g'(z) \;=\;\frac{1}{\sqrt{2\pi}}\exp\Bigl(-\tfrac12 z^2\Bigr)$} \\

& \ell(W) = \sum_{i=1}^n \left[\log |W| + \sum_{j=1}^d \log g'(w_j^T x^{(i)})\right] \\
&= \sum_{i=1}^n \log|W| + \sum_{j=1}^d\sum_{i=1}^n \log \left(\frac{1}{\sqrt
{2\pi}}\exp(-\frac{1}{2}w_j^T x^{(i)})^{2}\right) \\
&= \sum_{i=1}^n \log|W| + \sum_{j=1}^d\sum_{i=1}^n \left(-\frac{1}{2}\log(2\pi) + 
\log (-\frac{1}{2}w_j^T x^{(i)})^{2}\right) \\

&= n\log|W| - \frac{dn}{2}\log(2\pi) + \sum_{j=1}^d\sum_{i=1}^n (-\frac{1}{2}w_j^T x^{(i)})^{2} \\

& \text{Drop $\frac{dn}{2}\log(2\pi)$, Nothing to do with W} \\

&= n\log|W| - \sum_{j=1}^d\sum_{i=1}^n (\frac{1}{2}w_j^T x^{(i)})^{2} \\

& \nabla_W n\log|W| = n(W^{-1})^{T} \\

& \sum_{j=1}^d (w_j^\top x)^2 = \|W\,x\|^2 = (W x)^\top (W x) = x^\top W^\top W\,x
= \bigl(W\,x\,x^\top W^\top\bigr)
\\

& \nabla_W\;\tfrac12\,\|W x\|^2 = \tfrac12\;\nabla_W\, \bigl(Wxx^\top W^\top\bigr)
= \tfrac12\;2\,W\,x\,x^\top
= W\,x\,x^\top \\

& \nabla_W\Bigl(\tfrac12\sum_{i=1}^n\|W x^{(i)}\|^2\Bigr)
= \sum_{i=1}^n W\,x^{(i)}\,x^{(i)\top}
= W\;\Bigl(\sum_{i=1}^n x^{(i)} x^{(i)\top}\Bigr)
= W\,X^\top X \\

& \nabla_W \ell(W) = \nabla_W \left(n\log|W| \right) - \nabla_W \left(\sum_{j=1}^d\sum_{i=1}^n (\frac{1}{2}w_j^T x^{(i)})^{2} \right) \\

&= n(W^{-1})^{T} - WX^{T}X \\

& \text{Set $n(W^{-1})^{T} - WX^{T}X = 0$} \\
&= 0 
\\
& \text{We use: $W^{T}$ => $n(W^{-1})^{T} = WX^{T}X$}
\\
& n(W^{-1})^{T}W^{T} = W^{T}WX^{T}X \\
& nI = W^{T}WX^{T}X \\
& W^{T}W = n(X^{T}X)^{-1} \\

\\
& \text{We Know $C = X^\top X$ and $C = U\,\Lambda\,U^\top$, Whitening Matrix }\\
& W^\top W 
= (\sqrt{n}\,C^{-1/2})^\top(\sqrt{n}\,C^{-1/2})
= n\,(C^{-1/2}C^{-1/2})
= n\,C^{-1} \\
& \text{We Know Orthogonal Matrix $R^\top R = I$, Define $W' = R\,W$ } \\
& W'^\top W' = (R\,W)^\top (R\,W) = W^\top R^\top R\,W = W^\top W = n\,C^{-1}
\\
& \text{So, We have $W$ => $W^\top W = n\,C^{-1}$, $W'$ => $n\,C^{-1}$}

\end{aligned}
$$

### (b) Laplace source. derive the update rule for a single example in the form $W := W + \alpha (\ldots)$

$$
\begin{aligned}

& \text{We Know $f_L(s) = \frac{1}{2} \exp(-|s|)$ and $\ell(W) = \sum_{i=1}^n \left[\log |W| + \sum_{j=1}^d \log g'(w_j^T x^{(i)})\right]$} \\

& W := W + \alpha (\nabla_W \ell(W)) \\

& \nabla_W \ell(W) = \nabla_W (\log|W| - \sum_{j=1}^d \log \frac{1}{2}\exp(- |w_j^T x^{(i)}|)) \\

& \text{We Know $\log\bigl(\tfrac12\,e^{-\lvert w_j^\top x\rvert}\bigr)
= \log\tfrac12 \;-\;\lvert w_j^\top x\rvert$ Drop $\log\frac{1}{2}$} \\

& \text{$\frac{\partial}{\partial w_j}\;\lvert w_j^\top x\rvert
= sign(w_j^{T} x)\;x$} \\

& \nabla_W \sum_{j=1}^{d} |w^{T}_jx| = sign(Wx)x^{T} \\

&= (W^{-1})^{T} - sign(Wx)x^{T} \\

& W := W + \alpha ((W^{-1})^{T} - sign(Wx^{(i)})x^{(i)^{T}})

\end{aligned}
$$

### (c) Cocktail Party Problem(Code Problem...)

## 5. Markov decision processes

### (a) For any two finite-valued vectors $V_1$, $V_2$, Prove: $||B(V_1) - B(V_2)||_\infty \leq \gamma||V_1 - V_2||_\infty$, where $||V||_\infty = \max_{s \in S} |V(s)|$

$$
\begin{aligned}

& \text{We Know} \\
& ||B(V_1) - B(V_2)||_\infty = \max_{s \in S} |B(V_1)(s) - B(V_2)(s)| \\

& B(V_1)(s) - B(V_2)(s) = \max_{a \in A} \sum_{s' \in S} P_{sa}(s')[R_{sa}(s') + \gamma V_1(s')] - \max_{a \in A} \sum_{s' \in S} P_{sa}(s')[R_{sa}(s') + \gamma V_2(s')] \\

& \text{We have $\max_a f_a - \max_a g_a \leq \max_a (f_a - g_a)$}
\\
& B(V_1)(s) - B(V_2)(s) \leq \max_{a \in A} \sum_{s' \in S} P_{sa}(s')[\gamma V_1(s') - \gamma V_2(s')] \\

&= \gamma \max_{a \in A} \sum_{s' \in S} P_{sa}(s')[V_1(s') - V_2(s')] \\

& \text{We Know $\Bigl|\sum_{s'} a_{s'}\Bigr| \;\le\; \sum_{s'} |a_{s'}|$ and $c\ge0$ => $c\,|x|$}
\\
& \Bigl|\sum_{s'}P_{sa}(s')[V_1(s')-V_2(s')]\Bigr| \;\le\; \sum_{s'}P_{sa}(s')\,|V_1(s')-V_2(s')| \\
& \text{We have $\bigl|V_1(s')-V_2(s')\bigr| \;\le\; \|V_1 - V_2\|_\infty$} \\

& \text{Merge:}\\
& \Bigl|\sum_{s'}P_{sa}(s')[V_1(s')-V_2(s')]\Bigr| \;\le\; \sum_{s'}P_{sa}(s')\,\bigl|V_1(s')-V_2(s')\bigr| \;\le\; \sum_{s'}P_{sa}(s')\;\|V_1-V_2\|_\infty
\\
& \max_{s \in S} \left|B(V_1)(s) - B(V_2)(s)\right|\;\le\;\gamma\;\|V_1-V_2\|_\infty \\

\end{aligned}
$$

### (b) prove that B has at most one fixed point

$$
\begin{align*}

\text{Suppose we have two fixed points $U, V$ where $B(V)=V$ $B(U)=U$} \\

||V-U\||\infty = ||B(V)-B(U)||_\infty \le \gamma||V-U||_\infty \\

\text{$\gamma<1$, We have:} \\ 

||V-U||_\infty = 0 \\

\text{$V = U$ is only Solution}

\end{align*}
$$

## 6. Reinforcement Learning: The inverted pendulum(Code Problem...)