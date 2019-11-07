## Topic Modeling
A repo that reproduced multiple topic modeling algorithms from scratch  

### pLSA Probabilistic latent semantic analysis
https://arxiv.org/pdf/1301.6705.pdf

### Latent Dirichlet Allocation (LDA) algorithm - lda.py
The original paper could be found here http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf, the MCMC algorithm I used is the collapsed Gibbs Sampling https://people.cs.umass.edu/~wallach/courses/s11/cmpsci791ss/readings/griffiths02gibbs.pdf, which to me is simpler to implement than the variational inference proposed by the original author.
I test it on a small dataset, which includes around 120 Yelp reviews and covers three main categories (seafood, plumbing, pet shops) The algorithm could clearly identify the keywords for each topic.

### Biterm Topic Model for Short Texts algorithm - btm.py
The original paper could be found in http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf, 
the MCMC algorithm I used here is also collpased Gibbs Sampling


### Twitter Topic modeling
The original paper could be found in https://www.researchgate.net/publication/221397617_Comparing_Twitter_and_Traditional_Media_Using_Topic_Models


### Dataset
Yelp open sourced parts of the review data https://www.yelp.com/dataset/challenge
