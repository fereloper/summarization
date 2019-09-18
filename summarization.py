#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 12:40:37 2019

@author: feroj
"""
# Importing all the modules
import numpy as np
import tensorflow as tf
# import newspaper as ns
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dropout, Dense, LSTM, Activation
import os
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize

heading = {}
description = {}
heading = pickle.load(open('bbc_heading.pkl',"rb"))
description = pickle.load(open('bbc_description.pkl',"rb"))


for k, v in heading.items():
    heading[k] = v.decode('utf-8')

nltk.download('punkt')
all_text = []
for (i, j) in list(zip(description.values(), heading.values())):
    all_text.append(word_tokenize(i + j))

print(all_text[0])
vocab_list = [i.lower() for j in all_text for i in j]


def indexing(txt):
    """
    The function creates word to index mapping
    Param:
        txt: List of Vocab words
    """
    vocab = set(txt)
    vocab_to_idx = {v:k for k, v in enumerate(vocab)}
    idx_to_vocab = {v:k for k, v in vocab_to_idx.items()}
    return vocab, vocab_to_idx, idx_to_vocab

vocab, vocab_to_idx, idx_to_vocab = indexing(vocab_list)


# Prebuild files
with open("glove.6B.200d.txt","rb") as f:
    glove = f.readlines()


def glove_dict(glove_vector):
    """
    The function creates mapping between words and the GloVe vectors
    Param:
        glove_vector: List of GloVe words and their weights
    """
    word_weights = []
    for word in glove_vector:
        word_weights.append(word.split())
    
    print("Creating GloVe word and weight dictionary....")
    glove_words_weights = dict((i[0], i[1:]) for i in word_weights)
    
    print("Completed!")
    return glove_words_weights

glove_words_weights = glove_dict(glove)
pickle.dump(glove_words_weights, open('glove_words_weight.pkl','wb'))


# No of glove symbol
n_embeddings = 200
n_glove_symbols = len(glove_words_weights.keys())
print("Number of GloVe symbols: ", n_glove_symbols)
glove_weight_matrix = np.empty((n_glove_symbols, n_embeddings))
print(glove_weight_matrix.shape)
print(glove_weight_matrix[:10,:10])




c = 0
glove_index_dict = {}
global_scale = .1
for i in glove_words_weights.keys():
    w = i.decode("utf-8")
    glove_index_dict[w] = c 
    glove_weight_matrix[c,:] = glove_words_weights[i] 
    c += 1
    if c % 100000 == 0:
        print(c)
        
glove_weight_matrix *= global_scale
print("GloVe weight matrix std...", glove_weight_matrix.std())
print(glove_weight_matrix[:5,:5])


vocab_size = len(vocab)
shape = (vocab_size, n_embeddings)
scale = glove_weight_matrix.std() * np.sqrt(12/2)
print("Scale: ", scale)
embedding = np.random.uniform(low=-scale, high=scale, size=shape)
print("Embedding shape...",embedding.shape,"Std...",embedding.std())

c = 0
for i in range(vocab_size):
    w = idx_to_vocab[i]
    g = glove_index_dict.get(w, glove_index_dict.get(w))
    if g is not None:
        embedding[i,:] = glove_weight_matrix[g]
        c +=1

print("No. of tokens, found in GloVe matrix and copied to embedding...", c, float(c / vocab_size))


glove_threshold = 0.5
word2glove = {}
for w in vocab_to_idx:
    if w in glove_index_dict:
        g = w
        word2glove[w] = g

print("Word to glove lenght: ", len(word2glove))



normed_embedding = embedding/np.array([np.sqrt(np.dot(gweight,gweight)) for gweight in embedding])[:,None]

nb_unknown_words = 2600

glove_match = []
for w, idx in vocab_to_idx.items():
    if idx >= vocab_size-nb_unknown_words and w.isalpha() and w in word2glove:
        gidx = glove_index_dict[word2glove[w]]
        gweight = glove_weight_matrix[gidx,:].copy()
        
        # find row in embedding that has the highest cos score with gweight
        gweight /= np.sqrt(np.dot(gweight,gweight))
        score = np.dot(normed_embedding[:vocab_size-nb_unknown_words], gweight)
        while True:
            embedding_idx = score.argmax()
            s = score[embedding_idx]
            if s < glove_threshold:
                break
            if idx_to_vocab[embedding_idx] in word2glove :
                glove_match.append((w, embedding_idx, s)) 
                break
            score[embedding_idx] = -1
glove_match.sort(key = lambda x: -x[2])
print('# of glove substitutes found', len(glove_match))


for orig, sub, score in glove_match[-10:]:
    print(score, orig, '=>', idx_to_vocab[sub])

glove_idx2idx = dict((vocab_to_idx[w], embedding_idx) for  w, embedding_idx, _ in glove_match)

print(embedding.shape)
print(len(glove_match))

c = 0
for i in glove_idx2idx.keys():
    g = glove_idx2idx[i]
    embedding[i, :] = embedding[g]
    c += 1

print("Number of word substitued with the closest word is...", c)

# Building Encoder Network

max_length = 200
def model_builder(embeds):
    model = keras.Sequential()
    model.add(Embedding(weights=[embeds], name="embedding_1", input_dim=vocab_size,
                        output_dim=n_embeddings, input_length = max_length))
    for i in range(2):
        lstm = LSTM(rnn_size, name="layer_%s" %(i), return_sequences=True)
        model.add(lstm)
        model.add(Dropout(prob, name="drop_%s" %(i)))
        
    lstm = LSTM(rnn_size, name="layer_2", return_sequences=False)
    model.add(lstm)
    model.add(Dropout(prob, name="drop_2"))
    model.add(Dense(1))
    model.add(Activation('softmax', name="activation"))
    return model


rnn_size = 200
prob = 0.5
encoder = model_builder(embedding)
encoder.compile(loss='binary_crossentropy', optimizer='rmsprop')


desc_list = []
for i in description.values():
    desc_list.append(word_tokenize(i))

head_list = []
for i in heading.values():
    head_list.append(word_tokenize(i))

head_list[0]

doc2idx = []
for i in desc_list:
    doc2idx.append([vocab_to_idx[w.lower()] if w.lower() in vocab_to_idx.keys() else 0 for w in i])
print(len(doc2idx))


head2idx = []
for i in head_list:
    head2idx.append([vocab_to_idx[w.lower()] if w.lower() in vocab_to_idx.keys() else 0 for w in i])
    
print(head2idx[0])

padded_docs = pad_sequences(doc2idx, maxlen=max_length)

encoder.fit(padded_docs, head2idx, epochs=10, verbose=0)