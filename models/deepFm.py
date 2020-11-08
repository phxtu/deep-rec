#coding:utf-8
import tensorflow as tf
from tensorflow.keras import layers, Model

class DeepFm(object):
    def __init__(self, nfeats, nembs, mlps, useFm=True, useDeep=True, dropFm=None, dropDeeps=None, seed=0):
        self.nfeats = nfeats
        self.nembs = nembs
        self.mlps = mlps
        self.useFm = useFm
        self.useDeep = useDeep
        self.dropFm = dropFm if dropFm else [1,1]
        self.dropDeeps = dropDeeps if dropDeeps else [1]*(len(mlps+1))
        self.seed = seed
    
    def build(self):
        featSize = 39
        tf.random.set_random_seed(self.seed)
        inputs = {}
        for key in ['featI', 'featV']:
            inputs[key] = layers.Input(shape=(featSize,), name=key)
        # embedding
        embs = layers.Embedding(self.nfeats, self.nembs)
        firstOrdWgt = layers.Embedding(self.nfeats, 1)
        # ------------------- first order ---------------------
        firstOrd = tf.squeeze(firstOrdWgt(inputs['featI']), axis=-1) * inputs['featV']
        firstOrd = layers.Dropout(self.dropFm[0], name='dropFirstOrd')(firstOrd)
        # ------------------- second order ---------------------
        featEmbs = embs(inputs['featI']) * tf.expand_dims(inputs['featV'],-1)
        ## square of sum
        sum_squ = tf.square(tf.reduce_sum(featEmbs, axis=1))
        ## square sum
        squ_sum = tf.reduce_sum(tf.square(featEmbs), axis=1)
        ## second order
        secondOrd = (sum_squ - squ_sum) * 0.5
        secondOrd = layers.Dropout(self.dropFm[1], name='dropSecondOrd')(secondOrd)
        # ------------------- deep ---------------------
        deep = tf.reshape(featEmbs, [-1, featSize*self.nembs])
        deep = layers.Dropout(self.dropDeeps[0], name='dropDeep_0')(deep)
        for i,n in enumerate(self.mlps):
            deep = layers.Dense(n, activation='linear')(deep)
            # deep = layers.BatchNormalization(name='bn_%d'%i)(deep)
            deep = tf.nn.relu(deep)
            deep = layers.Dropout(self.dropDeeps[1+i], name='dropDeep_%d'%(i+1))(deep)
        # ------------------- deepFm ---------------------
        ## concat
        concats = []
        if self.useFm: concats.extend([firstOrd, secondOrd])
        if self.useDeep: concats.append(deep)
        concats = tf.concat(concats, -1) if len(concats)>1 else concats[0]
        output = layers.Dense(1, activation='sigmoid', name='pred')(concats)
        # ------------------- build model ----------------
        mdl = Model(inputs=inputs, outputs=output)
        return mdl
