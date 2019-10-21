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
unique_words = set()
for clean_review in clean_reviews:
    clean_review = clean_review.split()
    review_length = len(clean_review)
    cur_review_biterms = set()
    for i in range(review_length):
        unique_words.add(clean_review[i])
        # we use a interval of 5, if two words are disance to each other less than 5 positions, than count as a biterms
        for j in range(i+1, min(i+6, len(clean_review))):
            cur_review_biterms.add((clean_review[i], clean_review[j]))
    biterms.extend(list(cur_review_biterms))

# bigrams only
# biterms = [biterm for review in clean_reviews for biterm in zip(review.split(" ")[:-1], review.split("")[1:])]
# biterms = set(biterms)


def BTM(biterms, unique_words, num_of_topics, num_of_iterations):
    ####################################################################################
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
    n_z = np.random.randint(0, num_of_topics, N_BITERMS)
    # Words count over topics
    # Key is word, value is an array of topic counts, use the index to indicate the topic 1 to k
    n_wz = defaultdict(lambda: np.zeros(num_of_topics))
    for index, (w1, w2) in enumerate(biterms):
        n_wz[w1][n_z[index]] += 1
        n_wz[w2][n_z[index]] += 1

    # unlike to LDA model, in the biterm model, each bigram is coming from a specific topic
    # biterm_topic = np.zeros((N_BITERMS, num_of_topics))
    for iteration in range(num_of_iterations):
        print(iteration)
        for index, (w1, w2) in enumerate(biterms):
            cur_topic = n_z[index]
            # give a -1 class to the current biterm, means we ignore the current biterm
            # n_z[index] = -1
            n_wz[w1][cur_topic] -= 1
            n_wz[w2][cur_topic] -= 1

            # nz = np.unique(n_z, return_counts=True)[1][1:]
            nz = np.bincount(n_z, minlength=num_of_topics)
            nz[cur_topic] -= 1
            n_w1z = n_wz[w1]
            n_w2z = n_wz[w2]
#             print(n_w1z)
#             print(n_w2z)
#             print(nz)
            z_posterior = (nz + DL_ALPHA) * (n_w1z + DL_BETA) * (n_w2z + DL_BETA) / np.sum(
                (2 * nz + len(unique_words) * DL_BETA) * (2 * nz + len(unique_words) * DL_BETA))
            topic_prob = z_posterior / np.sum(z_posterior)
            topic_selection = np.argmax(
                np.random.multinomial(n=1, pvals=topic_prob, size=1))

            n_z[index] = topic_selection
            n_wz[w1][topic_selection] += 1
            n_wz[w2][topic_selection] += 1

    # return the topic assignment for each biterm
    return n_z

