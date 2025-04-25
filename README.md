# Simulation and Monte Carlo
## ABC for Alpha-Stable Models

---

### Question n°1: Validity of the CMS Method

To implement the method presented in Appendix B, the library **JAX** was used to improve performance.

Let's understand why the **Chambers–Mallows–Stuck (CMS) method** is valid:

Stable distributions are defined through their **characteristic functions**, because their density functions typically don't have a closed form (except for special cases like Gaussian, Cauchy, and Lévy).

The characteristic function for a standardized alpha-stable distribution is:

φ(t) = exp( -|t|^α * [1 + iβ * sgn(t) * tan(πα / 2)] ), for α ≠ 1


The CMS method samples from this distribution by transforming two random variables:

- U ~ Uniform(-π/2, π/2)
- W ~ Exponential(1)

Then compute:

X = [sin(αU) / cos(U)^(1/α)] * [cos((1−α)U) / W]^((1−α)/α)


This transformation generates a random variable with the same characteristic function as the target alpha-stable distribution (see proof in:  
**Rafał Weron, _On the Chambers–Mallows–Stuck method for simulating skewed stable random variables_, Statistics & Probability Letters, 1996**).

Behind this transformation is an **analytical inversion** of the characteristic function — constructing a transformation from known variables to the alpha-stable distribution.

This approach is guaranteed to be valid because:

1. The characteristic function fully defines the distribution.
2. The transformation matches the characteristic function exactly.

---

### Question n°2: RQMC vs MC for Sampling Alpha-Stable Laws

#### How to use RQMC

To generate `n` samples from an alpha-stable distribution, we need to sample:

- U ~ Uniform(-π/2, π/2)
- W ~ Exponential(1)

Instead of regular random sampling, we can use **RQMC** (Randomized Quasi-Monte Carlo) by generating a **scrambled Sobol sequence** in [0,1]^2.

Steps:

1. Generate (u, v) in [0,1]^2 using a Sobol sequence.
2. Transform:
   - u' = u * π - π/2
   - w = -log(v) (inverse CDF of Exponential(1))
3. Use the CMS method to generate alpha-stable samples from u' and w.

---

#### RQMC vs MC: Results

We aim to estimate E[cos(Y)] where Y is sampled from an alpha-stable distribution with parameters:  
(α, β, γ, δ) = (1.7, 0.5, 1, 0)

Results:

| Sampler        | Mean (E[cos(Y)]) | Std Dev | Execution Time (ms) |
|----------------|:----------------:|:-------:|:------------------:|
| RQMC (Sobol)   | 0.356             | 0.0011  | 50.5  ± 2.71         |
| MC             | 0.356             | 0.0094  | 47.1  ± 0.780          |

✅ As expected, **RQMC** significantly reduces the standard deviation compared to regular Monte Carlo.

---

### Question n°3: ABC Inference for Alpha-Stable Parameters

#### Summary Features

We implemented the summary statistics S2, S3, S4, S5 as described in the paper.

- Each function exists in both a **vectorized** and **non-vectorized** version.
- Initially, only non-vectorized functions were implemented, but the ABC algorithms were too slow, so we vectorized everything.

---

#### ABC Algorithms

##### Data Generation

The **observed data** were generated using the parameters:

(α, β, γ, δ) = (1.7, 0.9, 10, 10)

##### Prior Distributions

- α ~ Uniform(1.1, 2)
- β ~ Uniform(-1, 1)
- γ ~ Uniform(0, 300)
- δ ~ Uniform(-300, 300)

##### Epsilon Values

- For S2, S3, S4: epsilon = 0.78125
- For S5: epsilon = 100

*Note:* For S5, epsilon = 10 caused ABC-Reject to not finish even after 10 hours.

---

##### ABC-Reject Method

The method consists of:

1. Sampling θ from the prior.
2. Simulating data given θ.
3. Accepting θ if the simulated data is close enough to the observed data (within epsilon).

---

###### Results for ABC-Reject:

| Parameter | Mean  | 95% CI Low | 95% CI High | Statistic |
|-----------|:-----:|:----------:|:-----------:|:---------:|
| α (1.7)   | 1.740 | 1.455      | 1.986       | S1        |
| β (0.9)   | 0.056| -0.924     | 0.950       | S1        |
| γ (10)    | 10.379| 7.026      | 13.656      | S1        |
| δ (10)    | 9.759 | 8.123      | 11.781     | S1        |

*Observation*
Very good prediction, except for β which seems to be chosen randomly. 

| Parameter | Mean  | 95% CI Low | 95% CI High | Statistic |
|-----------|:-----:|:----------:|:-----------:|:---------:|
| α (1.7)   | 1.622 | 1.175      | 1.984       | S2        |
| β (0.9)   | -0.014| -0.935     | 0.953       | S2        |
| γ (10)    | 14.368| 6.357      | 26.832      | S2        |
| δ (10)    | 9.772 | 3.283      | 16.417      | S2        |

*Observation:*  
δ and γ are reasonably well estimated, but α and β estimates are widely spread across the prior.

| Parameter | Mean    | 95 % CI low | 95 % CI high | Statistic |
|-----------|:-------:|:-----------:|:------------:|----:      |
| α (1.7)   | 1.547   | 1.119       | 1.977        | S3        |
| β (0.9)   | -0.002  | −0.957      | 0.947        | S3        |
| γ (10)    | 184.981 | 13.228       | 295.651       | S3        |
| δ (10)    | 2.592    | -284.734       |  283.215      | S3        |

*Observation*
The result is even worth as it looks like all the parameters were taken randomly on the prior : epsilon must be to high

| Parameter | Mean    | 95 % CI low | 95 % CI high | Statistic |
|-----------|:-------:|:-----------:|:------------:|----:      |
| α (1.7)   | 1.674   | 1.155       | 1.986        | S4        |
| β (0.9)   | 0.082  | −0.976      | 0.956       | S4        |
| γ (10)    | 147.877 | 8.147       | 291.977       | S4        |
| δ (10)    | -5.377    | -280.375       |  282.740      | S4        |

*Observation*
Same conclusion as S3...

| Parameter | Mean    | 95 % CI low | 95 % CI high | Statistic |
|-----------|:-------:|:-----------:|:------------:|----:      |
| α (1.7)   | 1.672   | 1.184       | 1.991        | S5        |
| β (0.9)   | 0.045   | −0.951      | 0.949        | S5        |
| γ (10)    | 9.302   | 0.958       | 22.110      | S5        |
| δ (10)    | 10.485  | -7.009    |  28.198  | S5        |

*Observation* 
Here only β seems random.


###### Conlusion on ABC Reject

The running time of the algorithm was about 5 minutes for each summary statistic, thanks to the vectorized version. (In the non-vectorized version, the algorithm did not finish even after 2 hours.)

The results are quite good, especially considering that the chosen epsilon was relatively large to save computational time. However, for some statistics (notably S3 and S4), the parameters appear to have been selected almost randomly.

---

##### MCMC-ABC


The method consists of:

1. Proposing a new θ' based on the current θ using a proposal distribution with a random walk inside uniform distribution.
2. Simulating data of alpha stable data given θ'.
3. Accepting θ' with a probability that depends on how close the simulated data is to the observed data (within epsilon) and the prior and proposal distributions.


Because of the fact that delta and gammma parameters are widely spread, the random walk on the wide interval is less efficient compared to other methods.


`mcmc_abc_new.ipynb` and `helpers.ipynb` present the implementation and summary statistics.

### ABC-MCMC Algorithm

**Inputs:**
- Observed data: `y_obs`
- Prior distribution: `π(θ)`
- Proposal distribution: `q(θ' | θ)`
- Distance function: `d(y, y_obs)`
- Tolerance: `ε`
- Number of iterations: `N`

**Output:**
- Sequence of accepted θ values

**Algorithm:**

1. Sample initial `θ₀` from the prior `π(θ)`.
2. Simulate `y₀` from the model given `θ₀`.
3. If `d(y₀, y_obs) > ε`, repeat steps 1–2 until `d(y₀, y_obs) ≤ ε`.
4. Set `θ_current = θ₀`.
5. For `i = 1` to `N`:
    - Propose `θ'` ~ `q(θ' | θ_current)`.
    - Simulate `y'` from the model given `θ'`.
    - If `d(y', y_obs) ≤ ε`:
        - Compute acceptance probability:
          ```
          a = min(1, [π(θ') * q(θ_current | θ')] / [π(θ_current) * q(θ' | θ_current)])
          ```
        - Accept `θ'` with probability `a`:
            - If accepted, set `θ_current = θ'`.
    - Store `θ_current`.
6. Return all stored `θ` values.



 Epsilon Values change from one method to another proposing that S4 is one of the most efficient in terms of precision (epsilon from 0.05 to 0.2)

- For S3: epsilon = 2
- For S4: epsilon = from 0.05 to 0.2
- For S2, S1: epsilon = 100

---


###### Results for MCMC-ABC:


| Parameter | Mean    | 95 % CI low | 95 % CI high | Statistic |
|-----------|:-------:|:-----------:|:------------:|----:      |
| α (1.7)   | 1.474   | 1.118       | 1.982        | S2        |
| β (0.9)   | -0.399  | -0.981      | 0.381        | S2        |
| γ (10)    | 106.676 | 2.415       | 291.678      | S2        |
| δ (10)    | 3.670   | -115.552    | 109.175      | S2        |




| Parameter | Mean    | 95 % CI low | 95 % CI high | Statistic |
|-----------|:-------:|:-----------:|:------------:|----:      |
| α (1.7)   | 1.268   | 1.110       | 1.380        | S3        |
| β (0.9)   | -0.603  | -0.927      | -0.165       | S3        |
| γ (10)    | 105.503 | 21.503      | 174.260      | S3        |
| δ (10)    | 195.065 | 56.111      | 295.793      | S3        |


All the parameters are bad


| Parameter | Mean    | 95 % CI low | 95 % CI high | Statistic |
|-----------|:-------:|:-----------:|:------------:|----:      |
| α (1.7)   | 1.591   | 1.332       | 1.817        | S4        |
| β (0.9)   | 0.637   | 0.261       | 0.985        | S4        |
| γ (10)    | 156.562 | 1.328       | 291.665      | S4        |
| δ (10)    | -55.991 | -281.104    | 278.516      | S4        |



good alpha and beta but poor gamma delta

###### Conlusion on ABC Reject

The algorithm is time efficient for a reasonable number of iterations. It does estimate well the alpha and beta parameters. S4 is the statistic that minimised epsilon making it close to zero. However, problems arise with estimation of delta, gamma parameters, as their assignment is mostly random.



---
##### SMC-PRC-ABC

**SMC-PRC-ABC** combines two key ideas:

1. **Sequential tightening of epsilon:**  
   Start with a large epsilon (loose acceptance), then gradually reduce it by mutating accepted samples.
2. **Importance sampling:**  
   Particles are reweighted based on their likelihood.

Following Appendix A, the function was **not vectorized**, resulting in long computation times (about 1 hour per statistic).

---

###### Results for SMC-PRC-ABC:


| Parameter | Mean    | 95 % CI low | 95 % CI high | Statistic |
|-----------|:-------:|:-----------:|:------------:|----:      |
| α (1.7)   | 1.539   | 1.245       | 1.847        | S2        |
| β (0.9)   | -0.588  | −0.976      | 0.234        | S2        |
| γ (10)    | 28.118  | 17.183      | 39.411       | S2        |
| δ (10)    | 11.473  | 3.060       |  20.681      | S2        |

Quite a good delta and Gamma but alpha et beta are not very good

| Parameter | Mean    | 95 % CI low | 95 % CI high | Statistic |
|-----------|:-------:|:-----------:|:------------:|----:      |
| α (1.7)   | 1.556   | 1.125       | 1.972        | S3        |
| β (0.9)   | -0.020  | −0.939      | 0.937        | S3        |
| γ (10)    | 199.174 | 64.653      | 292.269      | S3        |
| δ (10)    | -21.951 | -282.838    |  288.510     | S3        |

All the parameters are bad


| Parameter | Mean    | 95 % CI low | 95 % CI high | Statistic |
|-----------|:-------:|:-----------:|:------------:|----:      |
| α (1.7)   | 1.682   | 1.604       | 1.769        | S4        |
| β (0.9)   | 0.728   | 0.414       | 0.985        | S4        |
| γ (10)    | 149.742 | 13.383      | 292.269      | S4        |                       
| δ (10)    | 2.859   | -280.892    | 288.510      | S4        |


A very good alpha and beta, but a bad gamma and delta



| Parameter | Mean    | 95 % CI low | 95 % CI high | Statistic |
|-----------|:-------:|:-----------:|:------------:|----:      |
| α (1.7)   | 1.590   | 1.139       | 1.977        | S5        |
| β (0.9)   | 0.288   | −0.719      | 0.951        | S5        |
| γ (10)    | 9.890   | 7.472       | 12.230       | S5        |
| δ (10)    | 8.961   | 6.079       |  12.273      | S5        |

very good delta and gamma but a bad alpha and beta

###### Conclusion on SMC-PRC-ABC

The running time of the algorithm was about 1 hour for each summary statistic. This is because we did not have time to vectorize the algorithm; otherwise, the running time would have been similar to, or even faster than, the other ABC algorithms. (In fact, it takes "only" 1 hour without vectorization, whereas the non-vectorized versions of the other algorithms did not even finish.)

The results are slightly better than with the other ABC algorithms, especially for the statistics S3 and S4.
---

### Final Observations

- For S2, delta and gamma are fairly well estimated, but alpha and beta still deviate significantly.
- For S4 and S5, alpha and beta estimates improve, while gamma and delta may deteriorate.

---

# Conclusion

- CMS is a valid method for simulating alpha-stable distributions.
- RQMC improves sampling efficiency and variance reduction.
- Summary statistics and careful epsilon tuning are key to accurate ABC inference.

