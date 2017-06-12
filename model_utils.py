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
    
def artm2blei():
    pass

def get_top_indices(target_values, N):
    order = np.argsort(target_values)[::-1]

    sorted_vals = target_values[order]
    ids = np.array(range(len(target_values)))
    ids = ids[order]
    return ids[:N]

def get_dict(dn):
    dictionary = artm.Dictionary()
    batch_vectorizer = None
    if len(glob.glob(os.path.join(dn, '*.batch'))) < 1:
        batch_vectorizer = artm.BatchVectorizer(data_path='', data_format='bow_uci', collection_name=dn, target_folder=dn)
    else:
        batch_vectorizer = artm.BatchVectorizer(data_path=dn, data_format='batches')


    if not os.path.isfile(dn + '/dictionary.dict'):
        dictionary.gather(data_path=batch_vectorizer.data_path)
        dictionary.save(dictionary_path=dn+'/dictionary.dict')

    dictionary.load(dictionary_path=dn+'/dictionary.dict')
    return batch_vectorizer, dictionary

    
def raw_phi2artm(initial_phi, phi_num2token, phi_tok2num, dictionary, regs, scores, topic_names, stopwords=None):
    num_topics = len(topic_names)
    model_name = 'pwt'
    model = artm.ARTM(topic_names=topic_names, 
                       scores=scores,
                       regularizers=[],
                       cache_theta=True)

    model.initialize(dictionary=dictionary)

    for reg in regs:
        model.regularizers.add(reg)
        
    protobuf_data, phi_numpy_matrix = model.master.attach_model("pwt")
    phi_numpy_matrix[:, :] = 0
    
    classes = getattr(protobuf_data, 'class_id')
    tokens = getattr(protobuf_data, 'token')
    data = zip(classes, tokens)

    bcg_base = 0
    for i, datum in enumerate(data):
        (class_, token) = datum
        if token in phi_tok2num:
            imported_id = phi_tok2num[token]
            phi_numpy_matrix[i, :num_topics] = initial_phi[imported_id, :]
        else:
            if token in stopwords:
                phi_numpy_matrix[i, num_topics] = 1
            else:
                phi_numpy_matrix[i, num_topics] = 0.01
    phi_numpy_matrix /= np.sum(phi_numpy_matrix, axis=0)
    #numpy.copyto(phi_numpy_matrix, initial_phi)

    return model, protobuf_data, phi_numpy_matrix


    
'''    
def tweak_phi(lda_phi, num2token, token2num, dn, regs, num_document_passes, num_outer_iterations, topic_names):
    docword = 'docword_{}.txt'.format(dn)
    data_path = os.path.join(os.path.abspath(os.getcwd()), docword)

    batch_vectorizer = get_batch_vectorizer(dn, data_path)
    dictionary = get_dict(dn, batch_vectorizer)

    scores = [
    artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary),
    artm.SparsityPhiScore(name='SparsityPhiScore'),
    artm.SparsityThetaScore(name='SparsityThetaScore'),
    artm.TopicKernelScore(name='TopicKernelScore', probability_mass_threshold=0.3),
    artm.TopTokensScore(name='TopTokensScore', num_tokens=10)
    ]

    model, topic_model, phi_numpy_matrix = prepare_model(lda_phi, num2token, token2num, dictionary, regs, scores, num_document_passes, topic_names)
    model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=num_outer_iterations)
    topic_model, phi_numpy_matrix = model.master.attach_model("pwt")

    return model, topic_model, phi_numpy_matrix
'''