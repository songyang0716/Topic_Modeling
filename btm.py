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
    DL_ALPHA = 1
    # constant we set for the LD prior (word distribution in a topic)
    DL_BETA = 0.01
    # Number of total biterms
    N_BITERMS = len(biterms)

    # Assign a random topic for each biterm
    n_z = np.random.randint(0, num_of_topics, N_BITERMS)
    n_topics = np.bincount(n_z, minlength=num_of_topics)

    # Words count over topics
    # Key is word, value is an array of topic counts, use the index to indicate the topic 1 to k
    n_wz = defaultdict(lambda: np.zeros(num_of_topics))
    for index, (w1, w2) in enumerate(biterms):
        n_wz[w1][n_z[index]] += 1
        n_wz[w2][n_z[index]] += 1

    # unlike to LDA model, in the biterm model, each bigram is coming from a specific topic
    # biterm_topic = np.zeros((N_BITERMS, num_of_topics))
    for iteration in range(num_of_iterations):
        for index, (w1, w2) in enumerate(biterms):
            #             cur_topic = n_z[index]
            n_wz[w1][n_z[index]] -= 1
            n_wz[w2][n_z[index]] -= 1

            n_topics[n_z[index]] -= 1
            n_w1z = n_wz[w1]
            n_w2z = n_wz[w2]

            z_posterior = np.zeros(num_of_topics)
#             z_posterior = (n_topics + DL_ALPHA) * (n_w1z + DL_BETA) * (n_w2z + DL_BETA) / np.sum(
#                 (2 * n_topics + len(unique_words) * DL_BETA) * (2 * n_topics + len(unique_words) * DL_BETA))
            for z in range(num_of_topics):
                z_posterior[z] = (n_topics[z] + DL_ALPHA) * (n_w1z[z] + DL_BETA) * (n_w2z[z] + DL_BETA) / np.sum(
                    (2 * n_topics[z] + len(unique_words) * DL_BETA) * (2 * n_topics[z] + len(unique_words) * DL_BETA))

            topic_prob = z_posterior / np.sum(z_posterior)
            topic_selection = np.argmax(
                np.random.multinomial(n=1, pvals=topic_prob, size=1))
            n_z[index] = topic_selection
            n_topics[topic_selection] += 1
            n_wz[w1][topic_selection] += 1
            n_wz[w2][topic_selection] += 1

    # return the topic assignment for each biterm and the topic distribution of each bigram
    return n_z, n_wz


# The topic distribution of the whole copus is
n_z, n_wz = BTM(biterms, unique_words, 3, 30)
DL_ALPHA = 1
topic_distribution = (np.bincount(n_z, minlength=num_of_topics) + DL_ALPHA) / (len(biterms) + num_of_topics * DL_ALPHA)

# The top words from each topics
n_wz_values = np.array([topic_freq for key, topic_freq in n_wz.items()])
n_wz_keys = [key for key, topic_freq in n_wz.items()]

DL_BETA = 0.01
wz = (n_wz_values + DL_BETA) / (np.sum(n_wz_values, axis=0) + len(unique_words) * DL_BETA)
for i in range(num_of_topics):
    print("for topic {}, the top words are: ".format(i))
    print(n_wz_keys[np.argsort(wz[:,i])[-10:][::-1]])

