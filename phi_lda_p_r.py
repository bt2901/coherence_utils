
import numpy as np
import os, glob, codecs
from coherence_lib import *
import itertools
from tqdm import tqdm 

T = 50

lda_phi = read_phi(15275, T)
    
num2token, token2num = read_vocab()

def read_theta(T, D):
    my_shape = (T, D)

    theta = numpy.zeros((T, D))
    doc = 0
    with open("doc_topic_proportion_for_topic_intrusion", "r") as f:
        for line in f:
            arr = line.strip().split(" ")
            for topic, prob in enumerate(arr):
                theta[topic, doc] = float(prob)
            doc += 1
    print doc
    theta /= np.sum(theta, axis=0)
    return theta

D = 200
theta = read_theta(50, D)
print np.sum(theta)
n_positions = 22331618
    
def theta_est(doc_id, topic):
    return 1.0/50
    


def calc_p_r():
    with open("out_pr_uniform.txt", "w") as f:
        for topic in range(T):
            total_len = 0
            covered_len = 0
            expected_covered_len = 0.0
            expected_words = 0.0

            print "topic {}".format(topic)
            #for target_values in (lda_phi[topic, :], blei_scores[topic, :], ): #LR_vector[topic, :]):
            for target_values in (lda_phi[topic, :], ):
                displayed_words, displayed_word_ids = top_words_in_topic(target_values, num2token)
                with codecs.open("docword_rtl-wiki.txt", "r", encoding="utf8") as fin:
                    for doc_id, line in enumerate(fin):
                        arr = line.split(" ")
                        for entry in arr:

                            splitted = entry.split(":")
                            word, count = splitted[0], int(splitted[-1])
                            if word in token2num:
                                total_len += count
                                word_id = token2num[word]
                                pwt = lda_phi[word_id, topic]
                                # TODO: better ptdw
                                ptdw = pwt / np.sum(lda_phi[word_id, :])
                                expected_words += count * ptdw
                                if word in displayed_words:
                                    #print word, count, ptdw, expected_words
                                    covered_len += count
                                    expected_covered_len += count * ptdw
                            else:
                                pwt = 0

                    
                #print "{}".format(displayed_words)
                '''
                print total_len, covered_len, float(covered_len) / total_len
                print expected_covered_len, expected_covered_len /  / total_len
                print total_len, expected_words, expected_words / total_len
                '''
                total_prob = lda_phi[displayed_word_ids, topic]
                total_prob = np.sum(total_prob)
                S = "prob(top_token|t) = {} total_prob = {} covered_len = {} expected_covered_len = {} expected_words = {}".format(expected_covered_len / expected_words, total_prob, covered_len, expected_covered_len, expected_words)
                f.write(S)
                print S
                
                
def calc_expected_words(T, words_data, local_theta, phi):
    expected_words = numpy.zeros((T, ))
    for word_id, count in words_data.items():
        pwt = lda_phi[word_id, :]
        # TODO: better ptdw
        ptdw = lda_phi[word_id, :] * local_theta[:]
        ptdw /= numpy.sum(ptdw)
        expected_words += count * ptdw

    return expected_words
                
def calc_covered_len(T, lda_phi, num2token, token2num):
    expected_covered_len = numpy.zeros((T, ))
    expected_words = numpy.zeros((T, ))
    displayed_word_ids = [set()] * T
    for topic in range(T):
        displayed_words, this_displayed_word_ids = top_words_in_topic(lda_phi[:, topic], num2token)
        displayed_word_ids[topic] = set(this_displayed_word_ids)
    with codecs.open("docword_rtl-wiki.txt", "r", encoding="utf8") as fin:
        for doc_id, line in tqdm(enumerate(fin), total=7838):
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
            expected_words += local_expected_words 
            for topic in range(T):
                for word_id in local_top_words[topic]:
                    count = words_data[word_id]
                    ptdw = lda_phi[word_id, :] * local_theta[:]
                    ptdw /= numpy.sum(ptdw)
                    expected_covered_len[topic] += count * ptdw[topic]
                    
    #print "{}".format(displayed_words)
    '''
    print total_len, covered_len, float(covered_len) / total_len
    print expected_covered_len, expected_covered_len /  / total_len
    print total_len, expected_words, expected_words / total_len
    '''
    print "expected words"
    print expected_words
    print "expected_covered_len"
    print expected_covered_len
    total_prob = numpy.zeros((T, ))
    for topic in range(T):
        topic_top_prob = lda_phi[list(displayed_word_ids[topic]), topic]
        total_prob[topic] = np.sum(topic_top_prob)
    print "total_prob"
    print total_prob
    print "proportion"
    print expected_covered_len / expected_words
    '''
    total_prob = np.sum(total_prob)
    S = "prob(top_token|t) = {} total_prob = {} covered_len = {} expected_covered_len = {} expected_words = {}".format(expected_covered_len / expected_words, total_prob, covered_len, expected_covered_len, expected_words)
    f.write(S)
    print S
    '''


def mark_word(window_id, index, marked_positions):
    
    (corpus_file, line_num, head_id, tail_id) = window_id
    word_position = (corpus_file, line_num, head_id + index)
    marked_positions.add(word_position)

def calc_word_count(words_in_window, topic_word_set, window_id, marked_positions):
    # relevant_indexes = [i for i, w in enumerate(words_in_window) if w in topic_word_set]
    # relevant_indexes = [i for i, w in enumerate(words_in_window) if w != "_"]
    relevant_indexes = {w: i for i, w in enumerate(words_in_window) if w != "_"}
    if len(relevant_indexes.keys()) >= 2:
        for i in relevant_indexes.values():
            mark_word(window_id, i, marked_positions)
        '''
        combs = itertools.combinations(relevant_indexes, 2)
        for i1, i2 in combs:
            #w1, w2 = words_in_window[i1], words_in_window[i2]
            #if w1 in topic_word_set and w2 in topic_word_set:
            mark_word(window_id, i1, marked_positions)
            mark_word(window_id, i2, marked_positions)
        '''
    
# TODO: DELETE STOP WORDS
# 
# 

def coherence_process_file(window_size, corpus_file, topic_word_set):
    marked_positions = set()
    # n_positions = 0
    #now process the corpus file and sample the word counts
    line_num = 0
    total_windows = 0

    for line in codecs.open(corpus_file, "r", "utf-8"):
        words = line.strip().split(" ")
        words = [(w if w in topic_word_set else "_")  for w in words]
        i=0
        doc_len = len(words)
        # n_positions += doc_len
        #number of windows
        if window_size != 0:
            num_windows = doc_len + window_size - 1
        else:
            num_windows = 1
        #update the global total number of windows
        total_windows += num_windows
        
        # we do not count coocurencies of same word
        words_types = set(words) - set("_")
        if len(words_types) < 2:
            continue
        for tail_id in range(1, num_windows+1):
            if window_size != 0:
                head_id = tail_id - window_size
                if head_id < 0:
                    head_id = 0
                words_in_window = words[head_id:tail_id]
            else:
                words_in_window = words
            window_id = (corpus_file, line_num, head_id, tail_id)
            calc_word_count(words_in_window, topic_word_set, window_id, marked_positions)

            i += 1

        line_num += 1
    return n_positions, marked_positions
    
def calc_coherence_stats(words):
    # calc by hand
    # maybe use a thing from old code
    # what they did with newlines?
    window_size = 20 #size of the sliding window;
    #dir_name = "raw_plaintexts_no_stop"
    dir_name = "raw_plaintexts"
    mask = os.path.join(dir_name, "*.txt")
    marked_positions = set()
    # n_positions = 0
    words_set = frozenset(words)
    for doc in tqdm(glob.glob(mask)):
        local_positions, local_marked_positions = coherence_process_file(window_size, doc, words_set)
        # n_positions += local_positions
        marked_positions |= local_marked_positions
        #if "000.txt" in doc:
        #    print doc
        #    print len(marked_positions), n_positions, float(len(marked_positions))/n_positions
    return n_positions, marked_positions

#calc_p_r()
#calc_covered_len(T, lda_phi, num2token, token2num)

'''        
unified_marked_positions = set()
global_positions = 0

with open("out_percent.txt", "w") as f:
    for topic in range(T):
        print "topic {}".format(topic)
        f.write("topic {}".format(topic))
        displayed_words, displayed_word_ids = top_words_in_topic(lda_phi[topic, :], num2token)
        print displayed_words
        n_positions, marked_positions = calc_coherence_stats(displayed_words)
        print "FINAL"
        print len(marked_positions), n_positions, float(len(marked_positions))/n_positions
        f.write("{} {} {}\n".format(len(marked_positions), n_positions, float(len(marked_positions))/n_positions))
        unified_marked_positions |= marked_positions
        print "current: {}".format(float(len(unified_marked_positions))/n_positions)
         
    print "OVERALL"
    print len(unified_marked_positions)
    print len(unified_marked_positions), n_positions, float(len(unified_marked_positions))/n_positions
    f.write("overall: {} {} {}\n".format(len(unified_marked_positions), n_positions, float(len(unified_marked_positions))/n_positions))
    '''

    
    
filtered_len = 0
actual_len = 0

with codecs.open("docword_rtl-wiki.txt", "r", encoding="utf8") as fin:
    for doc_id, line in tqdm(enumerate(fin), total=7838):
            arr = line.split(" ")
            words_counts = [(entry.split(":")) for entry in arr]
            words_data = {token2num[word]: int(count) for (word, count) in words_counts if word in token2num}
            actual_len += sum(int(x[1]) for x in words_counts)
            filtered_len += sum(words_data.values())
            
print actual_len, filtered_len
print float(filtered_len)/actual_len 
