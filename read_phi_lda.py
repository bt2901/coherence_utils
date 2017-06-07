import numpy 
import os, glob
#from coherence_lib import *

T = 50
stopwords = frozenset("""
i me my myself we our ours ourselves you your yours yourself yourselves
he him his himself she her hers herself it its itself they them their
theirs themselves what which who whom this that these those am is are
was were be been being have has had having do does did doing a an the
and but if or because as until while of at by for with about against
between into through during before after above below to from up down in
out on off over under again further then once here there when where why
how all any both each few more most other some such no nor not only own
same so than too very s t can will just don should now
""".split())


   
def calc_coherence(window, phi_numpy_matrix, dictionary, metric="L2"):
    res = 0
    actual_len = 0
    next_i, prev_i = 1, 0
    while prev_i < len(window) and window[prev_i] not in dictionary:
        next_i, prev_i = next_i + 1, prev_i + 1
    while next_i < len(window):
        prev, next = window[prev_i], window[next_i]
        prev_phi = get_phi_prob(phi_numpy_matrix, dictionary, prev)
        next_phi = get_phi_prob(phi_numpy_matrix, dictionary, next)
        delta = (prev_phi - next_phi)
        if metric == "L2":
            res += numpy.sum(delta * delta)
        elif metric == "L1":
            res += numpy.sum(numpy.abs(delta))
        elif metric == "argmax":
            i1, i2 = numpy.argmax(prev_phi), numpy.argmax(next_phi)
            res += numpy.sum(numpy.abs(delta[i1, i2]))
        actual_len += 1
        next_i, prev_i = next_i + 1, prev_i + 1
        while next_i < len(window) and window[next_i] not in dictionary:
            next_i, prev_i = next_i + 1, prev_i + 1
    
    return res / actual_len
        


def get_phi_prob(phi_numpy_matrix, dictionary, token, return_value = True):
    try:
        id = dictionary[token]
        return phi_numpy_matrix[id, :]
    except KeyError:
        bad_keys.add(token)
        if return_value:
            return numpy.ones(T)/T
        else:
            return None

def get_top_indices(target_values, N):
    order = numpy.argsort(target_values)[::-1]

    sorted_vals = target_values[order]
    ids = numpy.array(range(len(target_values)))
    ids = ids[order]
    return ids[:N]
def top_words_in_topic(target_values, num_2_token):

    order = numpy.argsort(target_values)[::-1]

    sorted_words = target_values[order]
    words_ids = numpy.array(range(len(target_values)))
    sorted_words_ids = words_ids[order]
    displayed_word_ids = [sorted_words_ids[i] for i in range(10)]
    #print num_2_token
    displayed_words = [num_2_token[id] for id in displayed_word_ids]
    
    return displayed_words, displayed_word_ids    

def calc_blei_scores(plsa_phi):
    '''
    score
    phi[wt] * [log(phi[wt]) - 1/T sum_k log(phi[wk])]
    '''
    T = plsa_phi.shape[0]
    blei_eps = 1e-100
    log_phi = numpy.log(plsa_phi + blei_eps)
    denom = numpy.sum(log_phi, axis=0)
    denom = denom[numpy.newaxis, :]
    
    score = plsa_phi * (log_phi - denom/T)
    return score
    
def read_phi(V, T):
    my_shape = (T, 15275)

    lda_phi = numpy.zeros((V, T))
    topic = 0
    with open("word-prob-in-topic", "r") as f:
        for line in f:
            arr = line.strip().split(" ")
            for i, prob in enumerate(arr):
                lda_phi[i, topic] = numpy.exp(float(prob))
            topic += 1
    return lda_phi


def read_vocab():
    num2token = {}
    token2num = {}
    with open("vocab", "r") as f:
        for i, line in enumerate(f):
            num2token[i] = line.strip()
            token2num[line.strip()] = i
    return num2token, token2num

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
    topical_profile = numpy.zeros(phi_numpy_matrix.shape[1])
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
    topical_profile = numpy.zeros(phi_numpy_matrix.shape[1])
    try:
        for next, prev in zip(window[1:], window[:-1]):
            prev_id, next_id = dictionary[prev], dictionary[next]
            topical_profile += phi_numpy_matrix[next_id, :]
            delta = (phi_numpy_matrix[prev_id, :] - phi_numpy_matrix[next_id, :])
            res += numpy.sum(delta * delta)

        topical_profile += phi_numpy_matrix[dictionary[window[0]], :]
        if topical_profile[topic] >= 0.02 * len(window):
            return res
        else:
            return None
    except KeyError:
        return None

        
window_size = 10
best_val, best_phrase = [float("inf")] * T, [""] * T
dir_name = "wiki_short"
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
                        print doc
                        #print topic
                        #print best_val[topic], 
                        #print best_phrase[topic]
                

def describe_topics_new(lda_phi, num2token, best_val, best_phrase):
    blei_scores = calc_blei_scores(lda_phi)
    #for topic in range(T):
    for topic in range(10):
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