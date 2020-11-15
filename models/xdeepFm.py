#coding:utf-8
import tensorflow as tf
from tensorflow.keras import layers, Model

class XDeepFm(object):
    def __init__(self, nfeats, nembs, mlps, cins=None, useFm=True, useDeep=True, dropFm=None, dropDeeps=None, seed=0):
        self.nfeats = nfeats
        self.nembs = nembs
        self.cins = cins
        self.mlps = mlps
        self.useFm = useFm
        self.useDeep = useDeep
        self.dropFm = dropFm if dropFm else [0,0]
        self.dropDeeps = dropDeeps if dropDeeps else [0]*(len(mlps+1))
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
        # ------------------- cin ---------------------
        featEmbs = embs(inputs['featI']) * tf.expand_dims(inputs['featV'],-1)
        c0 = tf.expand_dims(featEmbs, 2)
        ci = c0
        cins = [None]*len(self.cins)
        for i, c in enumerate(self.cins):
            # outer product: c_i, c0
            ci = tf.transpose(ci, [0,2,1,3]) * c0
            ''' there seems some problem for Conv3D in tf.keras 1.15.0
            # reduce_sum by cnn
            ci = tf.expand_dims(ci, -1)
            cnn = layers.Conv3D(c, ci.get_shape()[1:3]+[1], activation=None, data_format="channels_last", use_bias=False, name='cin_cnn_%d'%i)
            ci = tf.squeeze(cnn(ci), [2])
            ci = tf.transpose(ci, [0,3,1,2])
            '''
            # reduce_sum by Dense
            ci = tf.transpose(ci, [0,3,1,2])
            shp = ci.get_shape()
            ci = tf.reshape(ci, [-1, shp[1], shp[2]*shp[3]])
            ci = layers.Dense(c, activation=None, use_bias=False, name='cin_dnn_%d'%i)(ci)
            ci = tf.expand_dims(tf.transpose(ci, [0,2,1]), 2)
            # reduce sum along embedding dim
            cins[i] = tf.reduce_sum(ci, axis=[2,3], name='cin_%d'%i)
        cin = tf.concat(cins, -1, name='cin')
        # ------------------- deep ---------------------
        deep = tf.reshape(featEmbs, [-1, featSize*self.nembs])
        deep = layers.Dropout(self.dropDeeps[0], name='dropDeep_0')(deep)
        for i,n in enumerate(self.mlps):
            deep = layers.Dense(n, activation='linear')(deep)
            deep = layers.BatchNormalization(name='bn_%d'%i)(deep)
            deep = tf.nn.relu(deep)
            deep = layers.Dropout(self.dropDeeps[1+i], name='dropDeep_%d'%(i+1))(deep)
        # ------------------- deepFm ---------------------
        ## concat
        concats = []
        if self.useFm: concats.extend([firstOrd, cin])
        if self.useDeep: concats.append(deep)
        concats = tf.concat(concats, -1) if len(concats)>1 else concats[0]
        output = layers.Dense(1, activation='sigmoid', name='pred')(concats)
        # ------------------- build model ----------------
        mdl = Model(inputs=inputs, outputs=output)
        return mdl
