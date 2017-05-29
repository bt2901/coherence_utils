import numpy as np
import artm
from artm import score_tracker
import glob, os
import matplotlib.pyplot as plt

import model_utils

from palmettopy.palmetto import Palmetto
palmetto = Palmetto()

num_document_passes = 2
num_outer_iterations = 4
num_topics = 2
#num_outer_iterations = 10
num_document_passes = 2


model, batch_vectorizer, dictionary = model_utils.example_model()

model.initialize(dictionary=dictionary)
model.num_document_passes = num_document_passes
model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=num_outer_iterations)

print 'Perplexity: {}'.format(model.get_score('PerplexityScore').value)
    
print "Tokens:"
for t, topic_name in enumerate(model.topic_names):
    print topic_name + ': ',
    tokens = model.score_tracker['TopTokensScore'].last_tokens[topic_name]
    print tokens
    coherence = palmetto.get_coherence(tokens, coherence_type="cv")
    print 'Coherence: {}'.format(coherence)

coherence_values = np.zeros((num_topics, num_outer_iterations))
for iter in range(num_outer_iterations):
    all_tokens_on_iter = model.score_tracker['TopTokensScore'].tokens[iter]
    values = [palmetto.get_coherence(all_tokens_on_iter[t], coherence_type="cv") for t in model.topic_names]
    coherence_values[:, iter] = values

print coherence_values

print np.sum(coherence_values, axis=0)
y = np.sum(coherence_values, axis=0) / num_topics

plt.plot(y)
plt.show()