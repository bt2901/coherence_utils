#encoding=utf8
#model_utils.py

import numpy as np
import artm
import glob, os


def example_model():
    dn = 'kos'
    num_topics = 2
    num_outer_iterations = 10
    num_document_passes = 2

    batch_vectorizer = None
    if len(glob.glob(os.path.join('kos', '*.batch'))) < 1:
        batch_vectorizer = artm.BatchVectorizer(data_path='', data_format='bow_uci', collection_name=dn, target_folder=dn)
    else:
        batch_vectorizer = artm.BatchVectorizer(data_path=dn, data_format='batches')

    dictionary = artm.Dictionary()
    scores = [
    artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary),
    artm.TopTokensScore(name='TopTokensScore', num_tokens=10) # web version of Palmetto works only with <= 10 tokens
    ]

    if not os.path.isfile(dn + '/dictionary.dict'):
        dictionary.gather(data_path=batch_vectorizer.data_path)
        dictionary.save(dictionary_path=dn+'/dictionary.dict')

    dictionary.load(dictionary_path=dn+'/dictionary.dict')


    topic_names = ['topic_{}'.format(i) for i in range(num_topics)]
    model = artm.ARTM(topic_names=topic_names, 
                       scores=scores,
                       regularizers=[],
                       cache_theta=True)

    return model, batch_vectorizer, dictionary

def read_phi_blei():
    pass

def blei2artm():
    pass
    
def artm2blei():
    pass
