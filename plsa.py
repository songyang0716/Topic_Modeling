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

    # words distribution in each topics
    nwz = np.random.rand(num_of_unique_words, num_of_topics)
    pwz = nwz/nwz.sum(axis=0,keepdims=1)
    # the topic distribution for each document
    nzd = np.random.rand(num_of_topics, len(clean_reviews))
    pzd = nzd/nzd.sum(axis=0,keepdims=1)

    pzwd = np.zeros((num_of_topics, num_of_unique_words, len(clean_reviews)))

    for i in range(num_of_iterations):
    	# E-step
		pwd = np.matmul(pwz, pzd)
		for j in range(num_of_unique_words):
			for k in range(len(clean_reviews)): 
				pzwd[:,j,k] = np.multiply(pwz[j,:], pzd[:,k]) / pwd[j,k]

    	# M-step
    	











