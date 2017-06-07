#encoding=utf8
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

def read_phi_blei(V, T):
    # my_shape = (T, 15275)

    lda_phi = np.zeros((V, T))
    topic = 0
    with open("word-prob-in-topic", "r") as f:
        for line in f:
            arr = line.strip().split(" ")
            for i, prob in enumerate(arr):
                lda_phi[i, topic] = np.exp(float(prob))
            topic += 1
    return lda_phi
    
def read_vocab_blei():
    num2token = {}
    token2num = {}
    with open("vocab", "r") as f:
        for i, line in enumerate(f):
            num2token[i] = line.strip()
            token2num[line.strip()] = i
    return num2token, token2num


def blei2artm():
    pass
    
def artm2blei():
    pass

def get_top_indices(target_values, N):
    order = np.argsort(target_values)[::-1]

    sorted_vals = target_values[order]
    ids = np.array(range(len(target_values)))
    ids = ids[order]
    return ids[:N]
    
