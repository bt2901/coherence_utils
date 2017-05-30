
import numpy as np
import os, glob
from coherence_lib import *

T = 50

lda_phi = read_phi(15275, T)
    
num2token, token2num = read_vocab()





def describe_topics(lda_phi, num2token):
    blei_scores = calc_blei_scores(lda_phi)
    for topic in range(T):
        print "topic {}".format(topic)
        cur_words = set()
        for target_values in (lda_phi[topic, :], blei_scores[topic, :], ): #LR_vector[topic, :]):
            displayed_words, displayed_word_ids = top_words_in_topic(target_values, num2token)
            cur_words |= set(displayed_words)

        print "{}".format(cur_words)
            #print "Coherence = {}".format(measure_coherence(displayed_words))

bad_keys = set()


def top_N_topics(window, phi_numpy_matrix, dictionary, N=3):
    topical_profile = np.zeros(phi_numpy_matrix.shape[1])
    bad = 0
    for token in window:
        if token not in dictionary:
            bad += 1
        topical_profile += get_phi_prob(phi_numpy_matrix, dictionary, token)
    if bad > 5:
        return None
    order = numpy.argsort(topical_profile)[::-1]
    ordered_topics = numpy.array(range(T))[order]
    return ordered_topics[:N]
    


        
def calc_coherence_topical(window, phi_numpy_matrix, dictionary, topic):
    res = 0
    topical_profile = np.zeros(phi_numpy_matrix.shape[1])
    try:
        for next, prev in zip(window[1:], window[:-1]):
            prev_id, next_id = dictionary[prev], dictionary[next]
            topical_profile += phi_numpy_matrix[next_id, :]
            delta = (phi_numpy_matrix[prev_id, :] - phi_numpy_matrix[next_id, :])
            res += np.sum(delta * delta)

        topical_profile += phi_numpy_matrix[dictionary[window[0]], :]
        if topical_profile[topic] >= 0.02 * len(window):
            return res
        else:
            return None
    except KeyError:
        return None

        
window_size = 10
best_val, best_phrase = [float("inf")] * T, [""] * T
dir_name = "raw_plaintexts"
mask = os.path.join(dir_name, "*.txt")
'''
for doc in glob.glob(mask):
    with open(doc, "r") as f:
        for line in f:
            tokens = line.strip().split(" ")
            for i, token in enumerate(tokens):
                window = tokens[i: i+window_size]
                if len(window) < window_size:
                    continue
                #print window
                cur_val = calc_coherence(window, lda_phi, token2num)
                if cur_val is not None and cur_val > 0 and cur_val < best_val:
                    best_val, best_phrase = cur_val, window
                    print doc
                    print best_val, 
                    print best_phrase
'''                    



for doc in glob.glob(mask):
    with open(doc, "r") as f:
        for line in f:
            tokens = line.strip().split(" ")
            #window = tokens
            for i, token in enumerate(tokens):
                window = tokens[i: i+window_size]

                if len(window) < window_size:
                    continue
                if len(set(window) & stopwords) == 0:
                    continue
                top_topics = top_N_topics(window, lda_phi, token2num, N=3)
                if top_topics is None:
                    continue
                for topic in range(T):
                    if topic not in top_topics:
                        continue
                    cur_val = calc_coherence(window, lda_phi, token2num)
                    if cur_val is not None and cur_val > 0 and cur_val < best_val[topic]:
                        best_val[topic], best_phrase[topic] = cur_val, window
                        #print doc
                        #print topic
                        #print best_val[topic], 
                        #print best_phrase[topic]
                

def describe_topics_new(lda_phi, num2token, best_val, best_phrase):
    blei_scores = calc_blei_scores(lda_phi)
    for topic in range(T):
        print "topic {}".format(topic)
        cur_words = set()
        for target_values in (lda_phi[topic, :], blei_scores[topic, :], ): #LR_vector[topic, :]):
            displayed_words, displayed_word_ids = top_words_in_topic(target_values, num2token)
            cur_words |= set(displayed_words)

        print "{}".format(cur_words)
        print best_val[topic], 
        print best_phrase[topic]
            #print "Coherence = {}".format(measure_coherence(displayed_words))
describe_topics_new(lda_phi, num2token, best_val, best_phrase)