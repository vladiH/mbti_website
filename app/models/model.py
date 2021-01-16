#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


# In[ ]:


class PositionalEncoding(layers.Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_angles(self,pos, i, d_model):#pos:(seq_length,1), i:(1, d_model)
        angles = 1/(np.power(10000.,(2*(i//2))/d_model))#(1, d_model)
        return pos*angles #(seq_length, d_model)

    def call(self, inputs):
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        angles = self.get_angles(np.arange(seq_length)[:,np.newaxis],
                                np.arange(d_model)[np.newaxis,:],
                                d_model)#(seq_length, d_model)
        angles[:,0::2] = np.sin(angles[:,0::2])
        angles[:,1::2] = np.cos(angles[:,1::2])
        pos_encoding =  angles[np.newaxis,...]#(1, seq_length, d_model)
        return inputs + tf.cast(pos_encoding, tf.float32)#(batch, seq_length, d_model)


# In[ ]:


def scaled_dot_product_attetion(queries, keys, values, mask):
    """
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_k, depth_v)
        mask: Float tensor with shape broadcastable 
        """
    product = tf.matmul(queries, keys, transpose_b=True) # (..., seq_len_q, seq_len_k)

    keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_dot_product = product/ tf.math.sqrt(keys_dim)

    if mask is not None:
        scaled_dot_product += (mask*-1e9)

    attention_weights = tf.nn.softmax(scaled_dot_product,axis=-1)# (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, values) #(..., seq_len_q, depth_v)
    #q(1, 3)
    #k(4, 3)
    #v(4, 2)
    #result = (1,3).t((4,3)) => (1,4)*(4,2) =>(1,2)=(seq_len_q, depth_v)
    return output, attention_weights


# In[ ]:


class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model//self.num_heads

        self.wq = layers.Dense(self.d_model)
        self.wk = layers.Dense(self.d_model)
        self.wv = layers.Dense(self.d_model)

        self.top_dense = layers.Dense(self.d_model)

    def split_heads(self, inputs, batch_size):
        shape = (batch_size, -1, self.num_heads, self.depth)#(batch_size,seq_len,heads,depth)
        splited_inputs = tf.reshape(inputs, shape=shape)
        return tf.transpose(splited_inputs, perm=[0,2,1,3])#(batch_size,heads,seq_len,depth)

    def call(self, queries, keys, values, mask):
        batch_size = tf.shape(queries)[0]

        queries = self.wq(queries)# (batch_size, seq_len, d_model)
        keys = self.wk(keys)# (batch_size, seq_len, d_model)
        values = self.wv(values)# (batch_size, seq_len, d_model)


        queries = self.split_heads(queries, batch_size)# (batch_size, num_heads, seq_len_q, depth)
        keys = self.split_heads(keys, batch_size)# (batch_size, num_heads, seq_len_k, depth)
        values = self.split_heads(values, batch_size)# (batch_size, num_heads, seq_len_k, depth)

        attention, attention_weights = scaled_dot_product_attetion(queries, keys, values, mask)#(batch_size, num_heads, seq_len_q, depth)
        attention = tf.transpose(attention, perm=[0,2,1,3])#(batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(attention, shape=(batch_size,-1, self.d_model))#(batch_size, seq_len_q, d_model)
        outputs = self.top_dense(concat_attention)#(batch_size, seq_len_q, d_model)
        return outputs, attention_weights#(batch_size, seq_len_q, d_model),#(batch_size, num_heads, seq_len_q, seq_len_k)


# In[ ]:


class EncoderLayer(layers.Layer):
    def __init__(self,
                   ffn_units, num_heads, dropout):
        super(EncoderLayer,self).__init__()
        self.ffn_units = ffn_units
        self.num_heads = num_heads
        self.dropout = dropout

    def build(self, inputs_shape):
        self.d_model = inputs_shape[-1]
        self.multi_head_attention = MultiHeadAttention(self.num_heads)
        self.dropout_1 = layers.Dropout(rate = self.dropout)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)

        self.dense_1 = layers.Dense(units = self.ffn_units, activation="relu")
        self.dense_2 = layers.Dense(units = self.d_model)
        self.dropout_2 = layers.Dropout(rate = self.dropout)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask, training):
        #(batch_size, seq_len_q, d_model),#(batch_size, num_heads, seq_len_q, seq_len_k)
        attention, attention_weights = self.multi_head_attention(inputs,inputs,inputs, mask)
        attention = self.dropout_1(attention, training = training)
        attention = self.norm_1(attention+inputs)

        outputs = self.dense_1(attention)#(batch_size, seq_len_q, ffn_units)
        outputs = self.dense_2(outputs)#(batch_size, seq_len_q, d_model)
        outputs = self.dropout_2(outputs, training = training)
        outputs = self.norm_2(outputs)
        return outputs, attention_weights


# In[ ]:


class Encoder(layers.Layer):
    def __init__(self, nb_layers, 
                   ffn_units, 
                   num_heads, 
                   dropout, 
                   vocab_size,
                   d_model,
                   name="Encoder"):
        super(Encoder, self).__init__(name=name)
        self.nb_layers = nb_layers
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout)
        self.enc_layers = [EncoderLayer(ffn_units,
                                        num_heads, 
                                        dropout) for _ in range(nb_layers)]

    def call(self, inputs, mask, training):#(batch_size, seq_len_q)
        attention_weights = {}
        outputs = self.embedding(inputs)#(batch_size, seq_len_q, d_model)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)#(batch_size, seq_len_q, d_model)
        outputs = self.dropout(outputs, training=training)
        for i in range(self.nb_layers):
            outputs, block = self.enc_layers[i](outputs, mask, training)
            attention_weights['encoder_layer{}_block'.format(i+1)] = block
        return outputs, attention_weights#(batch_size, seq_len_q, d_model),#dict((batch_size, num_heads, seq_len_q, seq_len_k))


# In[ ]:


class Maxout(layers.Layer):
    def __init__(self, ln_unit):
        super(Maxout, self).__init__()
        self.ln_unit=ln_unit
        
    def call(self, x, axis =None):
        shape = x.get_shape().as_list()
        if shape[0] is None:
            shape[0] = -1
        if axis is None:  # Assume that channel is the last dimension
            axis = -1
        num_channels = shape[axis]
        if num_channels % self.ln_unit:
            raise ValueError('number of features({}) is not '
                             'a multiple of num_units({})'.format(num_channels, self.ln_unit))
        shape[axis] = shape[axis]//self.ln_unit
        shape += [self.ln_unit]#(batch, seq_len, d_model/ln_units, ln_units)
       
        outputs = tf.reduce_max(tf.reshape(x, shape), axis=-1)
        return outputs


# In[ ]:


class TopLayer(layers.Layer):
    def __init__(self, d_output, ln_units, dropout):
        super(TopLayer, self).__init__()
        self.maxout = Maxout(ln_units)
        self.dropout_1 = layers.Dropout(rate=dropout)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)

        self.dense_1 = layers.Dense(units = d_output)

    def call(self, inputs, training):
        outputs = self.maxout(inputs)
        outputs = self.dropout_1(outputs)
        outputs = layers.GlobalAveragePooling1D()(outputs)
        outputs = self.norm_1(outputs)
        outputs = self.dense_1(outputs)
        return outputs


# In[ ]:


class Transformer(tf.keras.Model):
    def __init__(self, vocab_size_enc,
                   d_output,
                   d_model,
                   nb_layers,
                   ffn_units,
                   ln_units,
                   num_heads,
                   dropout,
                   name="transformer"):
        super(Transformer, self).__init__(name = name)
        self.encoder = Encoder(nb_layers,
                              ffn_units,
                              num_heads,
                              dropout,
                              vocab_size_enc,
                              d_model)
        self.top = TopLayer(d_output, ln_units, dropout)

    def create_padding_mask(self, seq):
        mask = tf.cast(tf.math.equal(seq,0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :] 

    def create_look_ahead_mask(self,seq):
        seq_len = tf.shape(seq)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)),-1,0)
        return look_ahead_mask

    def call(self, enc_inputs, training):
        enc_mask = self.create_padding_mask(enc_inputs)
        enc_ouputs, attention_weights  = self.encoder(enc_inputs, enc_mask, training)
        top_outputs = self.top(enc_ouputs, 
                               training)
        return top_outputs, attention_weights 

