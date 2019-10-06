"""
(C) YANG SONG - 2019
Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation, as described in
Finding scientifc topics (Griffiths and Steyvers)
"""

# Import packages
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
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


def LDA(reviews, num_of_topics, num_of_iterations):
    ####################################################################################
    ### reviews: contains a list of reviews, and each review is a list of words      ###
    ### num_of_topics: number of topics to generate                                  ###
    ### number_of_iterations: collapsed gibbs sampling iterations                    ###
    ####################################################################################

    # constant we set for the LD prior (topic distributions in a document)
    DL_ALPHA = 0.5
    # constant we set for the LD prior (word distribution in a topic)
    DL_BETA = 0.01

    # topic assignment for each word
    word_topics = []
    # topic counts in each document, use the index to indicate topic
    ndk = np.zeros((len(reviews), num_of_topics))
    # topic count over each word, key is word, value is an array of topic counts, use the index to indicate the topic
    nkw = defaultdict(lambda: np.zeros(num_of_topics))

    # randomly initialize a topic for each words in the document
    for i, review in enumerate(reviews):
        topics = np.random.randint(0, num_of_topics, len(review))
        word_topics.append(topics)
    #    _, ndk[index] = np.unique(topics, return_counts=True)
        for j, word in enumerate(review):
            nkw[word][topics[j]] += 1
            ndk[i][topics[j]] += 1

    # iteration
    for i in range(num_of_iterations):
        if i % 100 == 0:
            print(i)
        for j, review in enumerate(reviews):
            for k, word in enumerate(review):
                nkw[word][word_topics[j][k]] -= 1
                ndk[j][word_topics[j][k]] -= 1
                # ignore the current topic settings
                # word_topics[j][k] = -1
                nk = np.sum(ndk, axis=0)

                # reset word_topic based on the posterior sampling
                topic_posterior = np.zeros(num_of_topics)
                # sample from p(z|.), topic_prob is the normalized posterior predictive distribution
                for z in range(num_of_topics):
                    topic_posterior[z] = (
                        ndk[j][z] + DL_ALPHA) * (nkw[word][z] + DL_BETA) / (nk[z])
                topic_prob = topic_posterior / np.sum(topic_posterior)
                topic_selection = np.argmax(
                    np.random.multinomial(n=1, pvals=topic_prob, size=1))
                word_topics[j][k] = topic_selection
                nkw[word][word_topics[j][k]] += 1
                ndk[j][word_topics[j][k]] += 1
    # return the latest topic assignment for each word
    return word_topics