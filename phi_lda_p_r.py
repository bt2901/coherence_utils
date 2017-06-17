
import numpy as np
import os, glob, codecs
from coherence_lib import *
import itertools
from tqdm import tqdm 

T = 50

lda_phi = read_phi(15275, T)
    
num2token, token2num = read_vocab()

D = 200
n_positions = 22331618



                
def calc_expected_words(T, words_data, local_theta, phi):
    expected_words = numpy.zeros((T, ))
    for word_id, count in words_data.items():
        pwt = lda_phi[word_id, :]
        # TODO: better ptdw
        ptdw = lda_phi[word_id, :] * local_theta[:]
        ptdw /= numpy.sum(ptdw)
        expected_words += count * ptdw

    return expected_words
                

def mark_word(window_id, index, marked_positions):
    
    (corpus_file, line_num, head_id, tail_id) = window_id
    word_position = (corpus_file, line_num, head_id + index)
    marked_positions.add(word_position)

# TODO: itertools.combinations(relevant_indexes, 2)
def calc_word_count(words_in_window, topic_word_set, window_id, marked_positions):
    # relevant_indexes = [i for i, w in enumerate(words_in_window) if w in topic_word_set]
    # relevant_indexes = [i for i, w in enumerate(words_in_window) if w != "_"]
    relevant_indexes = {w: i for i, w in enumerate(words_in_window) if w != "_"}
    if len(relevant_indexes.keys()) >= 2:
        for i in relevant_indexes.values():
            mark_word(window_id, i, marked_positions)

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
calc_covered_len(T, lda_phi, num2token, token2num)
    
    
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
