# Source
https://proceedings.neurips.cc/paper/2018/file/6a4d5952d4c018a1c1af9fa590a10dda-Paper.pdf

# Key points
- Function for converting an n-dimensional vector to a probability distribution over n objects.
- Framework for developing such functions easier and with more controll over sparsity
- Based on framework two functions are proposed: sparsegen-lin and sparsehourglass


# Solutions
## Sparsegen Activation Framework
- generic probability mapping function - $p(z) = sparsegen(z;g; \lambda) =\underset{p \in \Delta^{K-1}}{argmin}\mid \mid p-g(z) \mid \mid^2 - \lambda \mid \mid p \mid \mid^2$
- $g: \mathbb{R}^k \rightarrow \mathbb{R}^k$ - component wise function applied to z
- $\lambda < 1$ - controls the regularization strength
- sparsegen can be transformed to sparsemax: $sparsegen(z;g; \lambda) = sparsemax(\frac{g(z)}{1 - \lambda})$
- examples: 
	- for $g(z) = exp(z)$ and $\lambda = 1 - \sum_{j \in [K]} e^{z_j}$ function is softmax
	- for $g(z) = z$ and $\lambda = 0$ function is sparsemax
## Sparsegen-lin
- $p(z) = sparsegen(z; \lambda) =\underset{p \in \Delta^{K-1}}{argmin}\mid \mid p-z \mid \mid^2 - \lambda \mid \mid p \mid \mid^2$
- it applies L2 regulartisation to $p$ 
- with $\lambda$ it controls width of sparse region
## Sparsecon
Idea:
- take vector $q = (-q,\dots, -q) \in \mathbb{R}^k$ for  $q>0$ and connect it with point $z$ - point were such a line intersects with hyperplane $1^Tz=1$ - edge of simplex
- $g(z) = \alpha z + (1-\alpha)q$ where $\alpha = \frac{1+Kq}{\sum_i z_i + Kq}$ 
- $\lambda = 0$
- examples:
	- for $q=0$ it is sum normalisation - scale independent
	- for $g \rightarrow \infty$ it is sparsemax - translation independent
	- q in between - some tradeoff between  scale independent and translation independent
- cons:
	- for $\sum_i z_i < -Kq$ it is undefined
	- not Lipschitz continuous
## Sparsehourglass
- $p(z) = \underset{p \in \Delta^{K-1}}{argmin}\mid \mid p - \frac{1+Kq}{\mid\sum_i z_i\mid + Kq} z \mid \mid^2$
- for points satisfying $\sum_i z_i > 0$ - sparsecon
- other points are mapped to **mirror points** satisfying $\sum_i z_i > 0$
- **mirror points** has to follow:
	- $\sum_i z_i = -\sum_i \tilde{z}_i$
	- $\tilde(z_i) - \tilde(z_j) = z_i - z_j \forall_{i,j}$
	- $sparsehourglass(z) = sparsecone(\tilde{z})$
- Pros:
	- Lipschitz continuous
	- defined for all $z$
	- not scale independent and translation independent at the same time, but with proper $q$ can achive one of those
	


# Model



# References


