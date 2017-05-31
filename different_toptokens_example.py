import pickle 
import numpy
import matplotlib.pyplot as plt

import model_utils

from palmettopy.palmetto import Palmetto
palmetto = Palmetto()

def measure_coherence(displayed_words):
    displayed_words = displayed_words[-10:]
    result = palmetto.get_coherence(displayed_words, coherence_type="cv")
    return result

        
model, batch_vectorizer, dictionary = model_utils.example_model()

model.initialize(dictionary=dictionary)
model.num_document_passes = 1
model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)



def human_readable_dict(phi_data):
    tokens = getattr(phi_data, 'token')
    num_2_token = {i: tok for i, tok in enumerate(tokens)}
    return num_2_token

phi_data, plsa_phi = model.master.attach_model('pwt')
plsa_theta = model.get_theta()
num_2_token = human_readable_dict(phi_data)


def calc_weighted_pk(phi, theta):
    '''
    p(k) is prob(topic = k), defined as p(k) = sum_k n_k / n, 
    calculation of n_k is a bit tricky:
        n_t = sum_d sum_w n_wd p_tdw = sum_d theta_td
    alternatively:
        n_t = sum_w n_wt
            (where n_wt = sum_d n_wd p_tdw)
            
        (so I don't actually need theta here, but using it a bit more convenient)

    if we fix some word w, we can calculate weighted_pk:
    wp_k = p(k) p(w|k)
    '''
    n_k = numpy.sum(theta, axis=1)
    p_k = n_k / numpy.sum(n_k)
    
    weighted_pk = p_k[:, numpy.newaxis] * phi.transpose()
    return weighted_pk



def calc_ptw(phi, theta):
    weighted_pk = calc_weighted_pk(phi, theta)
    return weighted_pk / numpy.sum(weighted_pk, axis=0) # sum by all T

    
def calc_LR_vectorised(phi, theta):
    """
    Likelihood ratio is defined as
        L = phi_wt / sum_k p(k)/p(!t) phi_wk
    equivalently:
        L = phi_wt * p(!t) / sum_k!=t p(k) phi_wk
    after some numpy magic, you can get:
        L = phi[topic, id] * (1 - p_k[topic]) / {(sum(weighted_pk) - weighted_pk[topic])}
    numerator and denominator are calculated separately
    """

    weighted_pk = calc_weighted_pk(phi, theta)
            
    numerator = phi.transpose() * (1 - p_k[:, numpy.newaxis])
    denominator = (numpy.sum(weighted_pk, axis=0) - weighted_pk)
    #denominator[denominator < eps] = eps
    
    target_values = numerator / denominator
    target_values[denominator == 0] = float("-inf") # infinite likelihood ratios aren't interesting
    return target_values
    
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
    return score.transpose()
    


words_ids = numpy.array(range(len(num_2_token)))
LR_vector = calc_LR_vectorised(plsa_phi, plsa_theta)
blei_scores = calc_blei_scores(plsa_phi)
ptw_vector = calc_ptw(plsa_phi, plsa_theta)

methods = ["top", "lr", "blei"]

alpha_x = numpy.arange(0.1, 0.3, 0.01)
alpha_x = numpy.array([0.16, 1])
for alpha_val in alpha_x:
    methods.append ("ptw_alpha_{}".format(alpha_val))


coherence_avg = {m: 0 for m in methods}


for topic_index, topic_name in enumerate(model.topic_names):
    print "-------------"
    for description in methods:

        if description == "top":
            target_values = plsa_phi[:, topic_index]
        elif description == "lr": 
            target_values = LR_vector[topic_index, :]
        elif description == "blei": 
            target_values = blei_scores[topic_index, :]
        elif "ptw_alpha" in description: 
            alpha = float(description[10:])
            target_values = alpha * ptw_vector[topic_index, :] + (1-alpha) * plsa_phi[:, topic_index]
        else: 
            raise NameError


        displayed_word_ids = model_utils.get_top_indices(target_values, 10)
        displayed_words = [num_2_token[id] for id in displayed_word_ids]

        print "{}: {}".format(description, displayed_words)
        coh = measure_coherence(displayed_words)
        coherence_avg[description] += coh
        print "Coherence = {}".format(coh)
        
