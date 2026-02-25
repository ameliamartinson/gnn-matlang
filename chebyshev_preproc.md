Chebyshev / wavelet approximations are essentially ways to **avoid the eigendecomposition** in spectral methods by replacing the exact matrix function
[
\Phi(B)=U,\mathrm{diag}(\Phi(\lambda)),U^\top
]
with a **polynomial (or rational) approximation** that can be applied using only sparse matrix–vector/matrix–matrix multiplies.

That directly targets the bottleneck in **GNNML3’s preprocessing** (Algorithm 1), which currently does an eigendecomposition of (B) and then forms each initial support as (U\mathrm{diag}(\Phi_s(\lambda))U^\top) before masking and vectorizing into edge-features (C_0). 

---

## 1) Chebyshev polynomial approximation: what it is and why it’s fast

### 1.1 The basic setup: approximate a spectral filter without eigenvectors

For an undirected graph, the (normalized) Laplacian (L) is symmetric, so it has real eigenvalues and an orthonormal eigenbasis (the usual spectral GSP setting). The “Empirical Graph Wavelets” note describes spectral filtering and wavelets exactly in this form:
[
x_f = U \psi(\Lambda) U^\top x,
]
and wavelet coefficients at scale (s) as
[
W_x(s,n)=\big(U\psi(s\Lambda)U^\top x\big)_n. 
]

The problem is: diagonalizing (L) (or (B)) is too expensive for large graphs, so we want (\psi(L)x) **without** (U,\Lambda).

### 1.2 Scale the spectrum to ([-1,1])

Chebyshev polynomials (T_k(\cdot)) are numerically stable and have a simple recurrence on ([-1,1]). The wavelet note explicitly introduces:

* (T_n(x)=\cos(n\cos^{-1}x))
* recurrence (T_n(x)=2xT_{n-1}(x)-T_{n-2}(x)). 

So you rescale your operator (B) (e.g., (L) or normalized Laplacian) to (\tilde B) whose eigenvalues lie in ([-1,1]). For example, if (\lambda(B)\subseteq[0,\lambda_{\max}]), use:
[
\tilde B=\frac{2}{\lambda_{\max}}B - I.
]

(For normalized Laplacian, (\lambda_{\max}\le 2) is common; for adjacency you’d use a bound/estimate.)

### 1.3 Expand the filter in a truncated Chebyshev series

You approximate your desired filter (\psi(x)) by
[
\psi(x)\approx \sum_{k=0}^{K} c_k T_k(x).
]

The wavelet note gives the coefficient form for a Chebyshev expansion and explains why it converges well for smooth filters: with sufficient regularity, the coefficients decay like (O(n^{-k})), so relatively low (K) can work. 

### 1.4 Apply (\psi(\tilde B)) to a signal using only sparse multiplies

To compute (y=\psi(\tilde B)x), you never form (\psi(\tilde B)) as a dense matrix. Instead, you compute the Chebyshev vectors by recurrence:

* (t_0=x)
* (t_1=\tilde Bx)
* (t_k = 2\tilde B t_{k-1}-t_{k-2})

and then
[
y \approx \sum_{k=0}^{K} c_k t_k.
]

Each step needs one sparse multiply (\tilde B t_{k-1}). For features (H\in\mathbb{R}^{n\times f}), the same recurrence works column-wise, i.e. sparse–dense multiplies.

That’s the “fast localized spectral filtering” trick the wavelet note points to (and that ChebNet popularized): **replace eigendecomposition with repeated sparse multiplications**. 

---

## 2) How this relates to wavelets (and why “wavelet approximations” are basically “filter-bank approximations”)

A graph wavelet transform is not a different computational primitive—it's “just” spectral filtering with a **bank of scaled band-pass filters** (\psi(s\lambda)). The wavelet note defines it directly as (U\psi(s\Lambda)U^\top x). 

So the computational strategy is:

1. design a set of filters (scales (s) or bands) you care about;
2. approximate each (\psi_s(\cdot)) with a Chebyshev series;
3. apply each (\psi_s(\tilde B)) via the recurrence.

A key nuance: the note remarks that *plain Chebyshev approximation* “provides no clear methodology for controlling the band pass features of the filters,” motivating more tailored “empirical wavelet” constructions where you explicitly design filters supported over chosen spectral intervals. 

For GNNML3, this “filter-bank” viewpoint is very natural, because GNNML3 already uses **multiple spectral supports** (different (\Phi_s)) intended to cover different frequency bands. 

---

## 3) Can Chebyshev/wavelet approximations substitute GNNML3’s preprocessing step?

### 3.1 “In principle, yes”: GNNML3 supports are matrix functions, and those can be approximated

GNNML3’s initial supports are explicitly matrix functions of (B):
[
C_0^{(s)} = U\mathrm{diag}(\Phi_s(\lambda))U^\top
]
and the preprocessing extracts their entries on the mask (M) into (C_0). 

The paper itself notes that these spectral supports can be expressed as a **power series** of the Laplacian/adjacency (Theorem 7). 
Chebyshev expansions are just a *better-conditioned polynomial basis* than raw monomials (B^k), so conceptually they are very aligned with that observation.

So: **Chebyshev is absolutely applicable as a replacement for “eigendecomposition + exact spectral reconstruction.”**

### 3.2 “In practice, there are two different kinds of substitution”

The catch is what you actually need out of preprocessing.

#### Option A — Replace preprocessing by computing filtered messages on the fly (cleanest)

If you only need products like (C^{(s)}H), then you can avoid explicit (C_0) and do:

* choose (S) filters (\Phi_s),
* precompute Chebyshev coefficients ({c_{s,k}}_{k=0}^K),
* in each forward pass compute (Z_s \approx \Phi_s(\tilde B)H) via recurrence,
* feed (Z_s) into the rest of the layer.

This is basically “GNNML3-style multi-band filtering, but evaluated like ChebNet.”

This eliminates the expensive preprocessing step entirely and keeps runtime ~(O(SK|E|f)) per layer.

#### Option B — Keep GNNML3 architecture unchanged (edge-feature tensor (C_0)), but approximate (C_0)

GNNML3’s Algorithm 1 *does not* just apply supports to node features—it explicitly produces **edge features**:
[
C_0[:,s] = \text{sparse2vec}\Big( M \odot \Phi_s(B)\Big). 
]

To replicate that exactly, you need the **entries** ((i,j)) of (\Phi_s(B)) for ((i,j)\in\text{supp}(M)), not just (\Phi_s(B)H).

Chebyshev helps, but you still face a “selected entries of a matrix function” problem. Common strategies (conceptually) are:

* **Form a sparse polynomial approximation matrix** (p_s(B)) and then read off masked entries
  *Risk:* polynomial matrices can densify quickly as degree grows.
* **Stochastic probing / randomized trace-style estimators**: compute (p_s(B)z_r) for random vectors (z_r) using Chebyshev recurrence, and use outer-product sketches to estimate entries on (M).
  *Tradeoff:* introduces variance/noise in (C_0), but can be much cheaper than eigenvectors if (R) is small.

So yes, it *can* substitute preprocessing even in the “edge-feature” sense, but you typically end up with either:

* an approximation, or
* a modified architecture that no longer needs explicit (C_0).

---

## 4) Is this applicable beyond bipartite graphs?

Yes—**bipartite structure is not required** for Chebyshev/wavelet approximations.

Chebyshev polynomial filtering only needs that your operator (B) is something you can multiply by a vector efficiently (sparse), and that its spectrum can be scaled into ([-1,1]). For undirected graphs, the Laplacian (and normalized Laplacian) are symmetric, which is exactly the standard setting described in the wavelet note (undirected (G), Laplacian (L), spectral decomposition, filtering (U\psi(\Lambda)U^\top x)). 

In fact, the GNNML3 paper’s appendix explicitly describes ChebNet as relying on “the Chebyshev polynomial expansion of the scaled graph Laplacian,” tied to earlier spectral/wavelet ideas. 

So:

* **Non-bipartite, undirected graphs:** yes, fully applicable.
* **Directed or non-symmetric operators:** you need extra care (e.g., symmetrize, use a Hermitian construction, or use Arnoldi/Lanczos-type approximations). You didn’t ask about directed, but that’s the main boundary case.

---

## 5) Practical guidance: when is this a good substitute?

It tends to work best when:

* your desired (\Phi_s(\lambda)) are **smooth** (Gaussians like GNNML3’s (\exp(-b(\lambda-f_s)^2)) are smooth), because polynomial approximation converges faster for smooth functions (as noted via coefficient decay in the wavelet note). 
* you can tolerate **moderate polynomial degrees** (K) (otherwise you may lose the speed advantage),
* you prefer **streaming computation** (apply filters during training) instead of large one-time memory-heavy precomputes.

It’s harder when:

* you need very **sharp band-pass** behavior (high (K) needed, and the wavelet note explicitly flags band-pass control as tricky under pure Chebyshev approximations). 
* you must reproduce **explicit masked matrix entries** (Option B), not just filtered signals.

---

If you tell me which variant you’re aiming for—(A) “avoid eigendecomposition but okay changing the architecture” vs (B) “keep GNNML3 exactly but approximate (C_0)”—I can propose a concrete replacement pipeline (including how many probe vectors / polynomial degrees are typically needed and how to keep the approximation stable).
