# Source
https://arxiv.org/pdf/1707.08588v1.pdf

# Key points
- 'classic' softmax is slow - especially for large number of classes
- a self-organized hierarchical softmax is proposed
- this approach organizes the output vocabulary into a tree where the leaves are words and intermediate nodes are latent variables, or classes. The  tree structure could have many levels and there is a unique path from root to each word. The probability of a word is the product of probabilities of each node along its path.
# Solution
- two layer hierarchical softmax
- $\begin{bmatrix} h_c \\ h_w \end{bmatrix} = Relu \begin{bmatrix} W_c h \\ W_wh \end{bmatrix}$ where $W_c, W_h \in \mathbb{R}^{dxd}$ where $h$ is presoftmax hidden states
- cluster propability $P(c \mid h) = \frac{exp(h_c^TU_C^C)}{\sum exp(\sum_{c \in C}{exp(h_c^T U_C^C)})}$
- in-cluster propability $P(c \mid h, C(w_t)) = \frac{exp(h_w^T U_{w_t}^V)}{\sum exp(\sum_{c \in C(w_i)}{exp(h_w^T U_w^V)})}$
- $P(w_t \mid h) = P(c \mid h) * P(c \mid h, C(w_t))$
- loss functions:
	- cluster perplexity $ppl_{cluster}(C) = 2^{\frac{1}{M} \sum_{w_t} -log_2 p(C(w_t) \mid w_{<t})}$ - ahould be as low as possible 
	- in-cluster erplexity $ppl_{cluster}(C) = 2^{\frac{1}{M} \sum_{w_t} -log_2 p(C(w_t) \mid w_{<t}, C(w_t))}$ - should be as high as possible
- optimalisation alghorithm - greedy word clustering
# Results
- perplexity - almost same as full softmax
- training times - 3.5 times faster!
