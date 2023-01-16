# Source
https://arxiv.org/pdf/1602.02068.pdf

# Key points

- New function for mapping vector to propability space
- Similar to Softmax - with key improvment - it can produce  **sparse representation** 

# Decription

Article is decribing a new activation func-  
tion similar to the traditional softmax, but able  
to output sparse probabilities.

# Solutions

## Basic Softmax 
- $softmax_i(z)  = \frac{exp(z_i)}{\sum_{j}exp(z_j)}$
- It cannot produce sparse representation (containing 0's) unless $z_i \longrightarrow \infty$


## Sparsemax

- $sparsemax(z) := \underset{p \in \Delta^{K-1}}{argmin} \mid \mid p-z \mid \mid^2$ 
- it maps vector $z$ to $K-1$ dimensions simplex
- it is differentiable operation
- invariant to adding constant
- can produce sparse vector
- commutes with permutations - $p(Pz) = Pp(z)$ for any permutation matrix $P$

# Other contribiutions
## Classic logistic function for softmax - logistic loss (or negative log-likelihood
$L_{softmax}(z, k) = -log softmax_k(z) = -z_k + \sum_j exp(z_j)$
## Loss function for sparsemax
Function proposed by authors:
$L_{sparsemax}(z,k) = -z_k + \frac{1}{2} \sum_{j \in S(z)} (z_j^2 - \tau ^2(z)) + \frac{1}{2}$
where $\tau$ is treshhold value calculated from $z$ during projection on simplex.

### Function properties
- Differentiable and it's Jacobian is $\nabla_z L_{sparsemax}(z;k) = -\delta_k + sparsemax(z)$
- convex
- $L_{sparsemax}(z+c1;k) = L_{sparsemax}(z;k)$ dla $c \in  \mathbb\{R\}$ 

# Results 

Slightly better than softmax in some nlp application (classification and attention mechanisms)
# References

- Sparsemax implementation in PyTorch - simplex projection in $O(K log K)$ https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py 


