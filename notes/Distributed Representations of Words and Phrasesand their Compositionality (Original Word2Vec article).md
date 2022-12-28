# Source
https://arxiv.org/pdf/1310.4546v1.pdf

# Key points
- Skip-gram model for learning words vector representation
- Remarkamble speedup achieved through hierarchical softmax/negative sampling
- Model can organize concepts and learn implicitly the relationships between them (note exactly and with some limitations)

# Decription
Article is delivering a new method for learning multi-dimensional word representations. An objective was to train vectors that can capture a large number of precise syntactic and semantic word relationships.

# Solutions
- model : $p = \frac{1}{T} \sum_{t=1}^T \sum_{−c≤j≤c,j\neq0} logp(w_{t+j}\mid w_t)$
- Using skip gram model to train words embedding, but instead of use of classic softmax in the last layer, authors modified it and used techniue called hierarchical softmax. In softmax classifier, because of sum in denominator, the complexity is O(V), where V is corpus size- which can be millions in big models. Hierarchical model is O(K+1), where K is a constant associated with text corpus, but typically in [5, 20] range.
- As an alternative Negative Sampling can be used - in this method we draw randomly K negative samples from noise distribtion, complexity is the same:  O(K+1), where K is a constant associated with text corpus, but typically in [5, 20] range.
- Subsampling of Frequent Words - the propability of chosing a word during training is set to:
				$P(w_i) = 1-\sqrt{\frac{t}{f(w_i)}}$
	where $f(w_i)$ is is the frequency of word wi and t is a chosen threshold, typically around $10^{-5}$. Intuition behind: frequent words such as "in", "the" or "a" bring less information than rare ones.


# Model
Basic model used in tests
-  dimensions - 300
- window size - 5
- 692 000 tokens with count 5 or more
- 33 billions corpus size


# References
- https://aegis4048.github.io/demystifying_neural_network_in_skip_gram_language_modeling - good article explaining skip-gram architectures
- https://aegis4048.github.io/optimize_computational_efficiency_of_skip-gram_with_negative_sampling - article explaining hierarchical softmax in details

