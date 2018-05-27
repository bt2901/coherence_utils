# encoding=utf8
import pickle 
import pandas as pd
import numpy as np
import os, glob, codecs

import itertools
from tqdm import tqdm 
import pymorphy2
morph = pymorphy2.MorphAnalyzer()

def build_top_scores_dict(scores):
    top_scores_dict = {}
    for topic_name, row in scores.iterrows():
        top_scores = row.sort_values(ascending=False)[:10]
        top_scores_dict[topic_name] = top_scores
    return top_scores_dict

def get_toptokens_from_saved_model():
    pp = r"C:\Development\Github\intratext_fixes\sgm200\phi_word2"
    phi_good_pd = pd.read_pickle(pp).transpose()
    return build_top_scores_dict(phi_good_pd)




def is_form_of(w, all_toptokens):
    for h in morph.parse(w):
        if h.normal_form in all_toptokens:
            #print w, h.normal_form
            return h.normal_form
    return "_"

def build_forms(all_toptokens):
    normal_form_map = {}
    dir_name = r"C:\sci_reborn\postnauka\postnauka_clean"
    mask = os.path.join(dir_name, "*.txt")
    for doc in tqdm(glob.glob(mask)):
        for line in codecs.open(doc, "r", "utf-8"):
            words = line.strip().split(" ")
            for w in words:
                if w not in normal_form_map:
                    nf = is_form_of(w, all_toptokens)
                    normal_form_map[w] = nf
    return normal_form_map


top_scores_dict = get_toptokens_from_saved_model()

all_toptokens = frozenset([x 
        for topic in top_scores_dict.keys()
            for x in top_scores_dict[topic].index]) 


normal_forms = build_forms(all_toptokens)

with open("normalized_forms.pkl", 'wb') as pickle_file:
        pickle.dump(normal_forms, pickle_file, 2)

