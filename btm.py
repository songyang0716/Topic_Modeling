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

# extract all the biterms from the list, since in BTM model
# BTM directly models the word cooccurrence patterns based on biterms instead of documents
# A biterm denotes an unordered word-pair co-occurring in a short context
biterms = []



def BTM(reviews, num_of_topics, num_of_iterations):
    ####################################################################################
    ### reviews: contains a list of reviews, and each review is a list of words      ###
    ### num_of_topics: number of topics to generate                                  ###
    ### number_of_iterations: collapsed gibbs sampling iterations                    ###
    ####################################################################################

    # constant we set for the LD prior (topic distributions in a document)
    DL_ALPHA = 50 / num_of_topics
    # constant we set for the LD prior (word distribution in a topic)
    DL_BETA = 0.01










