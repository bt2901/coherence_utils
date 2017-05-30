
import glob
import os
#import matplotlib.pyplot as plt
import itertools

import numpy as np
import artm
from artm import score_tracker

num_topics = 50

sm_phi_tau = 0.01
sp_phi_tau = -0.01 / 2

smsp_theta_tau = -0.01 / 2
#decor_phi_tau = 100000000
decor_phi_tau = 0

sp_phi_tau = -0.00001

sp_phi_tau = -0.00001
decor_phi_tau = 1e3

smsp_theta_tau = 0

from coherence_lib import *

T = 50

lda_phi = read_phi(15275, T)
    
num2token, token2num = read_vocab()

num_document_passes = 5 # 1 2 NOT OK, 
num_outer_iterations = 4 # 4 OK, 5 NOT OK,

#num_document_passes = 5 # 5 OK, 1 NOT OK
#num_outer_iterations = 6 # 4 OK, 5 NOT OK


num_document_passes = 2 # 1 2 NOT OK, 
num_outer_iterations = 1 # 2 # 4 OK, 5 NOT OK,


#num_document_passes = 1 # 1 2 NOT OK, 
#num_outer_iterations = 0 # 4 OK, 5 NOT OK,

dictionary_name = 'dictionary'
pwt = 'pwt'
nwt = 'nwt'
rwt = 'rwt'
docword = 'docword_rtl-wiki_common.txt'





dn = "rtl-wiki"
data_path = os.path.join(os.path.abspath(os.getcwd()), docword)

batch_vectorizer = get_batch_vectorizer(dn, data_path)
dictionary = get_dict(dn, batch_vectorizer)


regs = [
artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=sp_phi_tau, topic_names=other_topics),
artm.SmoothSparsePhiRegularizer(name='SmoothPhi', tau=sm_phi_tau, topic_names=["background"]),
artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=decor_phi_tau)
]
scores = [
artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary),
artm.SparsityPhiScore(name='SparsityPhiScore'),
artm.SparsityThetaScore(name='SparsityThetaScore'),
artm.TopicKernelScore(name='TopicKernelScore', probability_mass_threshold=0.3),
artm.TopTokensScore(name='TopTokensScore', num_tokens=10)
]

other_topics = ['topic_{}'.format(i) for i in xrange(num_topics)]
topic_names = other_topics + ["background"]


#def run_experiment(initial_phi, regs, num_document_passes):

model_artm, attached_phi = prepare_model(lda_phi, num2token, token2num, dictionary, regs, scores, num_document_passes, topic_names)

model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=num_outer_iterations)


        
def get_toptokens(tt, topic):
    with open("tmp.txt", "w") as f:
        f.write(str(tt))
    ws = getattr(tt, 'weight')
    tokens = getattr(tt, 'token')
    topics = getattr(tt, 'topic_name')
    data = zip(ws, tokens, topics)
    for datum in data:
        w, cur_token, cur_topic = datum
        if topic == cur_topic:
            print "{} : {}".format(cur_token, w)
    print ""


def print_measures(model_plsa):
    print 'Sparsity Phi: {}'.format(
        model_plsa.get_score('SparsityPhiScore'))
    #print model_plsa.score_tracker['SparsityPhiScore'].value

    print 'Perplexity: {}'.format(
        model_plsa.get_score('PerplexityScore'))
    print model_plsa.get_score('PerplexityScore')
    
    print "Tokens:"
    for topic_name in model_plsa.topic_names[:2]:
        print topic_name + ': ',
        print ""
        get_toptokens(model_plsa.get_score('TopTokensScore'), topic_name)
    
def calc_perplexity(model):
    # model.scores.add(artm.PerplexityScore(name='perplexity'))
    model.transform(batch_vectorizer=batch_vectorizer, theta_matrix_type=None)
    return model.get_score('PerplexityScore')

calc_perplexity(model_artm)
print_measures(model_artm)

#topic_model, phi_numpy_matrix = model_artm.master.attach_model("pwt")
if num_outer_iterations:
    topic_model, attached_phi = model_artm.master.attach_model("pwt")
corrupt_phi(attached_phi)
calc_perplexity(model_artm)
print_measures(model_artm)

