"""
(C) YANG SONG - 2019
Implementation of the collapsed Gibbs sampler for
Biterm Topic Models, as described in
Biterm Topic Model for Short Texts (Yan,  Guo, Lan, Cheng)
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

# extract all the unique biterms from the reviews
# BTM directly models the word cooccurrence patterns based on biterms
# A biterm denotes an unordered unique word-pair co-occurring in a short context, each context in our example is a review
biterms = []
for clean_review in clean_reviews:
    clean_review = clean_review.split()
    review_length = len(clean_review)
    cur_review_biterms = set()
    for i in range(review_length):
        for j in range(i+1, review_length):
            cur_review_biterms.add((clean_review[i], clean_review[j]))
    biterms.extend(list(cur_review_biterms))

# bigrams only 
# biterms = [biterm for review in clean_reviews for biterm in zip(review.split(" ")[:-1], review.split("")[1:])]
# biterms = set(biterms)



def BTM(reviews, biterms, num_of_topics, num_of_iterations):
    ####################################################################################
    ### reviews: contains a list of reviews, and each review is a list of words      ###
    ### num_of_topics: number of topics to generate                                  ###
    ### number_of_iterations: collapsed gibbs sampling iterations                    ###
    ####################################################################################

    # constant we set for the LD prior (topic distributions in a document)
    DL_ALPHA = 50 / num_of_topics
    # constant we set for the LD prior (word distribution in a topic)
    DL_BETA = 0.01

    # Number of total biterms
    N_BITERMS = len(biterms)
    # Assign a random topic for each biterm
    biterm_topic = np.random.randint(0, num_of_topics, N_BITERMS)

    # unlike to LDA model, in the biterm model, each bigram is coming from a specific topic
    # biterm_topic = np.zeros((N_BITERMS, num_of_topics))
    for iteration in range(num_of_iterations):
        for index, biterm in enumerate(biterms):
            # give a -1 classes to the current biterm
            biterm_topic[index] = -1 
            nz = np.unique(biterm_topic, return_counts=True)[1][1:]

            z_posterior = (nz + DL_ALPHA)









