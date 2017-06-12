import numpy 
import os, glob, codecs
from model_utils import read_phi_blei, read_vocab_blei, get_top_indices, raw_phi2artm, get_dict
import itertools
from tqdm import tqdm 
import artm

# read Blei's model
dn = 'rtl-wiki_fromblei'
T = 50
vocab_size = 15275
blei_phi = read_phi_blei(vocab_size, T)
num2token, token2num = read_vocab_blei()

use_artm = True 
if use_artm: 
    # the results will be slightly different (different dictionaries) 
    # this is not important:
    # the purpose of this script is to show how to work with models in ARTM format
    batch_vectorizer, dictionary = get_dict(dn)
    topic_names = ["topic_{}".format(i) for i in range(T)]
    model, protobuf_data, phi_numpy_matrix = raw_phi2artm(blei_phi, num2token, token2num, dictionary, [], [], topic_names)

    lda_phi = numpy.array(model.get_phi())
    num2token = list(model.get_phi().index)
    token2num = {token: i for i, token in enumerate(num2token)}
else:
    lda_phi = blei_phi
    

def calc_expected_words(T, words_data, local_theta, phi):
    expected_words = numpy.zeros((T, ))
    for word_id, count in words_data.items():
        pwt = phi[word_id, :]
        ptdw = phi[word_id, :] * local_theta[:]
        ptdw /= numpy.sum(ptdw)
        expected_words += count * ptdw

    return expected_words

def get_words_data(line, token2num, displayed_word_ids):
    T = len(displayed_word_ids)
    arr = line.split(" ")
    words_counts = [(entry.split(":")) for entry in arr]
    words_data = {token2num[word]: int(count) for (word, count) in words_counts if word in token2num}
    local_words = set(words_data.keys())
    local_top_words = [displayed_word_ids[topic] & local_words for topic in range(T)]

    return local_top_words, words_data

def calc_local_theta(phi, local_top_words, words_data):
    T = len(local_top_words)
    total_len = sum(words_data.values())
    local_theta = numpy.ones((T, )) / T
    for i in range(5):
        local_expected_words = calc_expected_words(T, words_data, local_theta, phi)
        local_theta = local_expected_words / total_len
        
    return local_theta, local_expected_words
    
def get_displayed_words_for_every_topic(T, lda_phi, num2token):
    displayed_word_ids = [set()] * T
    for topic in range(T):
        top_word_ids = get_top_indices(lda_phi[:, topic], 10)
        displayed_words = [num2token[id] for id in top_word_ids]
        displayed_word_ids[topic] = set(top_word_ids)
    return displayed_word_ids


def calc_outside_prob(T, lda_phi, token2num, displayed_word_ids):
    expected_inside_words = numpy.zeros((T, ))
    expected_outside_words = numpy.zeros((T, ))
    expected_surprise_outside_words = numpy.zeros((T, ))
    with codecs.open("docword_rtl-wiki.txt", "r", encoding="utf8") as fin:
        for doc_id, line in tqdm(enumerate(fin), total=7838):
            local_top_words, words_data = get_words_data(line, token2num, displayed_word_ids)
            local_theta, local_expected_words = calc_local_theta(lda_phi, local_top_words, words_data)

            for topic in range(T):
                for word_id in local_top_words[topic]:
                    count = words_data[word_id]
                    ptdw = lda_phi[word_id, :] * local_theta[:]
                    ptdw /= numpy.sum(ptdw)
                    expected_inside_words[topic] += count * ptdw[topic]
                    for diff_topic in range(T):
                        if diff_topic != topic:
                            expected_outside_words[topic] += count * ptdw[diff_topic]
                            if word_id not in local_top_words[diff_topic]:
                                expected_surprise_outside_words[topic] += count * ptdw[diff_topic]
                            
    print "number of top words related to this topic:" 
    print expected_inside_words
    print "number of top words unrelated to this topic:"
    print expected_outside_words
    print "...not even in top words of another topic:"
    print expected_surprise_outside_words
    print "prob(topic is k|word is in top words of topic k):"
    print expected_inside_words/(expected_outside_words + expected_inside_words)
    
    return expected_inside_words, expected_outside_words

displayed_word_ids = get_displayed_words_for_every_topic(T, lda_phi, num2token)
calc_outside_prob(T, lda_phi, token2num, displayed_word_ids)
