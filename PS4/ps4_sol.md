# PS4-Solution

### 1. Neural Networks: MNIST image classification(code Problem...)
...

### 2. Oﬀ Policy Evaluation And Causal Inference

##### (a) Importance Sampling
Prove: if $\hat{\pi}_0 = \pi_0$, $E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} R(s, a)$ = $E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)$

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

##### (b) Weighted Importance Sampling
Prove: if $\hat{\pi}_0 = \pi_0$, $\frac{E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{\hat{\pi}_0(s,a)} R(s, a)}{E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{\hat{\pi}_0(s,a)}}$ = $E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)$

$$
\begin{align*}
\text{Same Reason, We Get:} \\
\frac{E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{\hat{\pi}_0(s,a)} R(s, a)}{E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{\hat{\pi}_0(s,a)}} &= \frac{E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{{\pi}_0(s,a)} R(s, a)}{E_{s \sim p(s), a \sim \pi_0(s,a)} \frac{\pi_1(s,a)}{{\pi}_0(s,a)}} \\
&= \frac{\frac{\pi_1(s,a)}{\pi_0(s,a)} \sum_{(s,a)} R(s, a)p(s)\pi_0(s, a)} {\frac{\pi_1(s,a)}{\pi_0(s,a)} \sum_{(s,a)} p(s)\pi_0(s, a)} \\
&= \frac{\sum_{(s,a)} R(s, a)p(s) \pi_1(s,a)} {\sum_{(s,a)} p(s) \pi_1(s,a)} \\
&= E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a) \\

\end{align*}
$$

##### (c) 

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

##### (d) Doubly Robust

**i**
Prove: if $\hat{\pi}_0 = \pi_0$, $E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} \hat{R}(s, a)) + \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} (R(s, a) - \hat{R}(s, a))\right)$ = $E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)$

$$
\begin{align*}

E_{s \sim p(s), a \sim \pi_0(s,a)}\left(E_{s \sim p(s), a \sim \pi_1(s,a)} \hat R(s, a)\right) = E_{s \sim p(s), a \sim \pi_1(s,a)} \hat R(s, a)

\\
E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} \hat{R}(s, a)) + \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} (R(s, a) - \hat{R}(s, a))\right)
&= E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} \hat{R}(s, a)) + \frac{\pi_1(s, a)}{{\pi}_0(s, a)} (R(s, a) - \hat{R}(s, a))\right) \\
&= E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} \hat{R}(s, a)) + (\frac{\pi_1(s, a)}{{\pi}_0(s, a)} R(s, a) - \frac{\pi_1(s, a)}{{\pi}_0(s, a)}  \hat{R}(s, a))\right) \\
&= E_{s \sim p(s), a \sim \pi_0(s,a)} \left(E_{a \sim \pi_1(s,a)} \hat{R}(s, a) + (E_{ a \sim \pi_1(s,a)} R(s, a) - E_{a \sim \pi_1(s,a)} \hat R(s, a))\right) \\
&= E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)


\end{align*}
$$

**ii**
Prove: if $\hat{R}(s, a) = R(s, a)$, $E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} \hat{R}(s, a)) + \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} (R(s, a) - \hat{R}(s, a))\right)$ = $E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)$

$$
\begin{align*}
E_{s \sim p(s), a \sim \pi_0(s,a)}\left(E_{s \sim p(s), a \sim \pi_1(s,a)} \hat R(s, a)\right) = E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)

\\
E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} \hat{R}(s, a)) + \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} (R(s, a) - \hat{R}(s, a))\right)
&= E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} {R}(s, a)) + \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} (R(s, a) - {R}(s, a))\right) \\
&= E_{s \sim p(s), a \sim \pi_0(s,a)} \left((E_{a \sim \pi_1(s,a)} {R}(s, a)) + (\frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)} R(s, a) - \frac{\pi_1(s, a)}{\hat{\pi}_0(s, a)}  {R}(s, a))\right) \\
&= E_{s \sim p(s), a \sim \pi_0(s,a)} \left(E_{a \sim \pi_1(s,a)} {R}(s, a) + (E_{ a \sim \pi_1(s,a)} R(s, a) - E_{a \sim \pi_1(s,a)} R(s, a))\right) \\
&= E_{s \sim p(s), a \sim \pi_1(s,a)} R(s, a)


\end{align*}
$$

##### (e)

**i**<br>
- Importance Sampling Estimator, $R(s,a)$ too complicated, Not applicable to regression estimators.


**ii**<br>
- Regression estimators, $\hat \pi$ too complicated, Not applicable to Importance Sampling Estimator.

### 3. Principal components analysis(PCA)

$$
\begin{align*}

\end{align*}
$$

### 4. Independent components analysis(ICA)

##### (a) Gaussian source

$$
\begin{align*}

\end{align*}
$$

##### (b) Laplace source

$$
\begin{align*}

\end{align*}
$$

##### (c) Cocktail Party Problem

$$
\begin{align*}

\end{align*}
$$

### 5. Markov decision processes

##### (a)

$$
\begin{align*}

\end{align*}
$$

##### (b)

$$
\begin{align*}

\end{align*}
$$

### 6. Reinforcement Learning: The inverted pendulum
