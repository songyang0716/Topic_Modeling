"""
(C) YANG SONG - 2019
Implementation of the EM algorithm for
pLSA model, as described in
Probabilistic Latent Semantic Analysis (Thomas Hofmann)
https://arxiv.org/pdf/1301.6705.pdf
"""

# Import packages
import pandas as pd
import numpy as np
import datetime
from collections import defaultdict
import nltk
import random

stemmer = nltk.stem.snowball.SnowballStemmer("english")
data_dir = "/Users/yangsong/Desktop/Projects/Topic_Modeling/reviews_small.txt"
np.random.seed(666)

# read review texts
reviews = []
f = open(data_dir, "r")
for review in f:
	reviews.append(review)
random.shuffle(reviews)

# process text
# tokenize, lower, remove stop words, stem, then only keep alphabets in the string
clean_reviews = []
for review in reviews:
	s = nltk.word_tokenize(review)
	s = [word.lower() for word in s]
	s = [word for word in s if not word in set(
		nltk.corpus.stopwords.words('english'))]
	s = [stemmer.stem(word) for word in s if word.isalpha()]
	clean_reviews.append(s)

unique_words = set([word for review in clean_reviews for word in review])

def plsa(clean_reviews, num_of_topics, num_of_iterations, num_of_unique_words):
	####################################################################################
	### clean_reviews: clean reviews that has been tokenized, store in a list        ###
	### num_of_topics: number of topics to generate                                  ###
	### number_of_iterations: collapsed gibbs sampling iterations                    ###
	####################################################################################
	# words dictionary
	word_index = {}
	index_word = {}
	index = 0
	for review in clean_reviews:
		for word in review:
			if word in word_index:
				pass
			else:
				word_index[word] = index
				index_word[index] = word
				index += 1

	# words counts matrix
	n_doc = len(clean_reviews)

	# record the count of each word occured in each document
	ndw = np.zeros((n_doc, num_of_unique_words))
	for d, review in enumerate(clean_reviews):
		for word in review:
			l = word_index[word]
			ndw[d, l] += 1

	# words distribution in each topics
	nwz = np.random.rand(num_of_unique_words, num_of_topics)
	pwz = nwz/nwz.sum(axis=0,keepdims=1)
	# the topic distribution for each document
	nzd = np.random.rand(num_of_topics, n_doc)
	pzd = nzd/nzd.sum(axis=0,keepdims=1)

	pzwd = np.zeros((num_of_topics, num_of_unique_words, n_doc))


	for i in range(num_of_iterations):
		# E-step
		for w in range(num_of_unique_words):
			for d in range(len(clean_reviews)): 
				### very very important condition !
				if np.dot(pwz[w,:], pzd[:,d]) == 0:
					pzwd[:,w,d] = np.zeros(num_of_topics)
				else:
					pzwd[:,w,d] = np.multiply(pwz[w,:], pzd[:,d]) / np.dot(pwz[w,:], pzd[:,d])

		# M-step
		# update pwz
		for k in range(num_of_topics): 
			for w in range(num_of_unique_words):
				pwz[w,k] = np.matmul(ndw[:,w], pzwd[k,w,:])
			### very very important condition !
			if np.sum(pwz[:,k]) == 0:
				pwz[:,k] = np.zeros(n_doc)
			else:
				pwz[:,k] = pwz[:,k] / np.sum(pwz[:,k])


		# update pzds
		for d in range(n_doc):
			for k in range(num_of_topics):
				pzd[k,d] = np.matmul(ndw[d,:], pzwd[k,:,d]) 
			### very very important condition !
			if np.sum(pzd[:,d]) == 0:
				pzd[:,d] = np.zeros(num_of_unique_words)
			else:
				pzd[:,d] =  pzd[:,d] / np.sum(pzd[:,d])

	return pwz, pzd, index_word