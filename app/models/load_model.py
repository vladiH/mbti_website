#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os, sys
PATH = os.path.dirname(os.path.abspath(__file__))
if '__file__' in globals():
    sys.path.append(PATH)

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from model import Transformer
from preprocessing import process_text
# In[ ]:


DICTIONARY = [{0:'I', 1:'E'},{0:'S', 1:'N'},{0:'T', 1:'F'},{0:'J', 1:'P'}]

# In[ ]:


class LoadModel:
    def __init__(self):
        self.MAX_LENGTH_OUTPUT = 4
        self.D_MODEL = 256 #512
        self.NB_LAYERS = 8 #6
        self.FFN_UNITS = 2048 #2048
        self.LN_UNITS = 4
        self.NB_HEADS = 8 #8
        self.DROPOUT = 0.1 #0.1
        
    def initModel(self):
        return Transformer(vocab_size_enc = self.VOCAB_SIZE_INPUT,
               d_output = self.MAX_LENGTH_OUTPUT,
               d_model = self.D_MODEL,
               nb_layers = self.NB_LAYERS,
               ffn_units = self.FFN_UNITS,
               ln_units = self.LN_UNITS,
               num_heads = self.NB_HEADS,
               dropout = self.DROPOUT)

    def tokenizer(self):
        url = os.path.join(PATH,'./data/tokenizer_inp')
        return tfds.deprecated.text.SubwordTextEncoder.load_from_file(url)
    
    def loadWeight(self):
        try:
            url = os.path.join(PATH,'./data/weight/weight')
            self.model.load_weights(url)
        except:
            print("Some error happend!!")
            
    def load(self):
        self.tokenizer_inp = self.tokenizer()
        self.VOCAB_SIZE_INPUT = self.tokenizer_inp.vocab_size + 2
        self.model = self.initModel()
        self.loadWeight()
        
    def evaluate(self, inp_sentence, thresh=0.5):
        inp_sentence = [self.VOCAB_SIZE_INPUT-2]+self.tokenizer_inp.encode(inp_sentence)+ [self.VOCAB_SIZE_INPUT-1]
        enc_input = tf.expand_dims(inp_sentence, axis=0)
        #enc_input = tf.keras.preprocessing.sequence.pad_sequences(enc_input, maxlen=150, padding='post',value=0)
        predictions,_ = self.model(enc_input,  False)
        values = tf.nn.sigmoid(predictions)
        predictions = tf.cast(tf.greater(values, thresh), tf.float32)
        return predictions[0], values
    
    def predict(self, sentence):
        sentence = process_text(sentence)
        predictions, rates = self.evaluate(sentence)
        predictions= np.array(predictions.numpy(), dtype=np.int0)
        predicted_sentence = [mbti[id] for id, mbti in zip(predictions, DICTIONARY)]
        return predicted_sentence, rates

