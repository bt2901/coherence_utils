
import numpy 
import os, glob, codecs
from model_utils import read_phi_blei, read_vocab_blei, get_top_indices
import itertools
from tqdm import tqdm 
import artm



T = 50
vocab_size = 15275
lda_phi = read_phi_blei(vocab_size, T)
    
num2token, token2num = read_vocab_blei()
                
def calc_expected_words(T, words_data, local_theta, phi):
    expected_words = numpy.zeros((T, ))
    for word_id, count in words_data.items():
        pwt = lda_phi[word_id, :]
        ptdw = lda_phi[word_id, :] * local_theta[:]
        ptdw /= numpy.sum(ptdw)
        expected_words += count * ptdw

    return expected_words

def get_stuff(line, token2num, displayed_word_ids):
    T = len(displayed_word_ids)
    arr = line.split(" ")
    words_counts = [(entry.split(":")) for entry in arr]
    words_data = {token2num[word]: int(count) for (word, count) in words_counts if word in token2num}
    local_words = set(words_data.keys())
    local_top_words = [displayed_word_ids[topic] & local_words for topic in range(T)]
    total_len = sum(words_data.values())
    local_theta = numpy.ones((T, )) / T
    for i in range(5):
        local_expected_words = calc_expected_words(T, words_data, local_theta, lda_phi)
        local_theta = local_expected_words / total_len
        
    return local_theta, local_expected_words, local_top_words, words_data

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
        for doc_id, line in tqdm(enumerate(fin)):
            local_theta, local_expected_words, local_top_words, words_data = get_stuff(line, token2num, displayed_word_ids)

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
                            
    print "expected_inside_words"
    print expected_inside_words
    print "expected_outside_words"
    print expected_outside_words
    print "expected_surprise_outside_words"
    print expected_surprise_outside_words
    print "prob(t|top_tok)"
    print expected_inside_words/(expected_outside_words + expected_inside_words)
    return expected_inside_words, expected_outside_words

displayed_word_ids = get_displayed_words_for_every_topic(T, lda_phi, num2token)
calc_outside_prob(T, lda_phi, token2num, displayed_word_ids)

raise NotImplementedError


dn = "rtl-wiki_fromblei"

decor_phi_tau = 1e7
regs = [
#artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=sp_phi_tau, topic_names=other_topics),
#artm.SmoothSparsePhiRegularizer(name='SmoothPhi', tau=sm_phi_tau, topic_names=["background"]),
artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=decor_phi_tau)
]
num_topics = 50
topic_names = ['topic_{}'.format(i) for i in xrange(num_topics)]

num_document_passes = 5
num_outer_iterations = 5

num_document_passes = 1
num_outer_iterations = 1

model, tt, attached_phi = tweak_phi(lda_phi, num2token, token2num, dn, regs, num_document_passes, num_outer_iterations, topic_names)


def convert_phi(topic_model, phi_numpy_matrix, num_topics, vocab_size, target_class):

    result = numpy.zeros((vocab_size, num_topics))
    classes = getattr(topic_model, 'class_id')
    
    
    tokens = getattr(topic_model, 'token')
    data = zip(classes, tokens)
    num2token, token2num = {}, {}
    my_token_id = 0
    print phi_numpy_matrix.shape
    print len(data)
    for i, datum in enumerate(data):
        (class_, token) = datum
        if class_ == target_class:
            num2token[my_token_id] = token
            token2num[token] = my_token_id
            result[my_token_id] = phi_numpy_matrix[i, :]
            my_token_id += 1
    return result, num2token, token2num


numpy_phi, num2token, token2num = convert_phi(tt, attached_phi, T, vocab_size, u"@default_class")

displayed_word_ids = get_displayed_words_for_every_topic(T, numpy_phi, num2token)
calc_outside_prob(T, numpy_phi, token2num, displayed_word_ids)
