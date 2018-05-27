# encoding=utf8
import pickle 
import pandas as pd
import numpy as np
import os, glob, codecs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors

import itertools
from tqdm import tqdm 
import pymorphy2

morph = pymorphy2.MorphAnalyzer()
my_cmap = colors.LinearSegmentedColormap.from_list("", ["skyblue", "red", "green"])

def build_top_scores_dict(scores):
    top_scores_dict = {}
    for topic_name, row in scores.iterrows():
        top_scores = row.sort_values(ascending=False)[:10]
        top_scores_dict[topic_name] = top_scores
    return top_scores_dict

def get_toptokens_from_saved_model(pp):
    phi_good_pd = pd.read_pickle(pp).transpose()
    return build_top_scores_dict(phi_good_pd)



def mark_word(window_id, index, marked_positions):
    (corpus_file, line_num, head_id, tail_id) = window_id
    word_position = (corpus_file, line_num, head_id + index)
    marked_positions.add(word_position)

def calc_word_count(words_in_window, window_id, marked_positions):
    relevant_indexes = [i for i, w in enumerate(words_in_window) if w != "_"]
    words_here = {w for w in words_in_window if w != "_"}
    if len(words_here) >= 2:
        for i in relevant_indexes:
            mark_word(window_id, i, marked_positions)


def coherence_process_file(window_size, corpus_file, topic_word_set):
    marked_positions = set()
    interesting_positions = set()
    pos2line = {}
    n_positions = 0
    #now process the corpus file and sample the word counts
    total_windows = 0

    line_num = 0
    for line in codecs.open(corpus_file, "r", "utf-8"):
        pos2line[line_num] = n_positions
        words = line.strip().split(" ")
        words = [normal_forms[w] for w in words]
        words = [(w if w in topic_word_set else "_") for w in words]
        doc_len = len(words)
        if doc_len <= 1:
            continue
        n_positions += doc_len
        line_num += 1
        interesting_positions |= set( [(line_num, j, w) for j, w in enumerate(words) if w != "_"])

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
            calc_word_count(words_in_window, window_id, marked_positions)

    pos2line[line_num] = n_positions
    return n_positions, marked_positions, interesting_positions, pos2line
    
def calc_coherence_stats(words, dir_name):
    window_size = 20 #size of the sliding window;
    #dir_name = "raw_plaintexts_no_stop"
    dir_name = r"C:\sci_reborn\postnauka\postnauka_clean"
    mask = os.path.join(dir_name, "*.txt")
    marked_positions = set()
    n_positions = 0
    words_set = frozenset(words)
    for doc in tqdm(glob.glob(mask)):
        local_positions, local_marked_positions, local_int_positions, pos2line = coherence_process_file(window_size, doc, words_set)
        n_positions += local_positions
        marked_positions |= local_marked_positions

    return n_positions, marked_positions

def collect_doc_lines(local_int_positions, local_marked_positions, pos2line, y):
    doc_lines = [[], ]
    for line_num in sorted(pos2line.keys()[:-1]):
        len_line = pos2line[line_num+1] - pos2line[line_num]
        pic_arr = list(np.random.random(len_line) * 0.1)
        rest = (y - (len_line % y))
        pic_arr += [np.nan] * (rest)
        doc_lines.append(pic_arr)

    for entry in local_int_positions:
        (line_num, index, w) = entry
        doc_lines[line_num][index] = 0.5
    for entry in local_marked_positions:
        (corpus_file, line_num, index) = entry
        doc_lines[line_num][index] = 1
    return doc_lines

def show_document_colorized(doc_lines, y):
    x = sum(len(a) for a in doc_lines) / y
    pic = np.array( [val for arr in doc_lines for val in arr] )
    plt.imshow(pic.reshape(x, y), interpolation='none', cmap=my_cmap)
    # draw legend
    rect_nan = patches.Rectangle((0,0),1,1,facecolor='w')
    rect_word = patches.Rectangle((0,0),1,1,facecolor=my_cmap(0.1))
    rect_top = patches.Rectangle((0,0),1,1,facecolor=my_cmap(0.5))
    rect_rep = patches.Rectangle((0,0),1,1,facecolor=my_cmap(1.0))
    plt.legend(
       (rect_nan, rect_word, rect_top, rect_rep), 
       ('end of paragraph', 'word', 'top token', 'represented token'), 
       handlelength=1, handleheight=1,
       loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout(pad=7)
    plt.show()

def show_picture_of_marks(words, dir_name):
    window_size = 20 #size of the sliding window;
    mask = os.path.join(dir_name, "*.txt")
    marked_positions = set()
    n_positions = 0
    words_set = frozenset(words)
    for doc in tqdm(glob.glob(mask)):
        local_positions, local_marked_positions, local_int_positions, pos2line = coherence_process_file(window_size, doc, words_set)
        if local_positions > 200 and len(local_marked_positions) > 3:
            y = int(window_size * 2)
            doc_lines = collect_doc_lines(local_int_positions, local_marked_positions, pos2line, y)

            show_document_colorized(doc_lines, y)


if __name__ == "__main__":
    unified_marked_positions = set()

    phi_file = r"C:\Development\Github\intratext_fixes\sgm200\phi_word2"
    dir_name = r"C:\sci_reborn\postnauka\postnauka_clean"
    top_scores_dict = get_toptokens_from_saved_model(phi_file)

    with open("normalized_forms.pkl", 'rb') as pickle_file:
        normal_forms = pickle.load(pickle_file)

    should_plot = not True
    should_calc_stats = not False

    if should_plot:
        for topic in top_scores_dict.keys():
            displayed_words = top_scores_dict[topic].index
            show_picture_of_marks(displayed_words, dir_name)

    if should_calc_stats:
        with codecs.open("out_percent.txt", "w", "utf8") as f:
            f.write("topic name;marked_positions;total_positions;fraction\n")
            for topic in top_scores_dict.keys():
                f.write(u"topic {};".format(topic))
                displayed_words = top_scores_dict[topic]        
                displayed_words = displayed_words.index

                n_positions, marked_positions = calc_coherence_stats(displayed_words, dir_name)
                f.write("{};{};{}\n".format(len(marked_positions), n_positions, float(len(marked_positions))/n_positions))
                unified_marked_positions |= marked_positions
                 
            f.write("overall;{};{};{}\n".format(len(unified_marked_positions), n_positions, float(len(unified_marked_positions))/n_positions))
