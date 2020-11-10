import tensorflow as tf
import tensorflow.keras.backend as K
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.36
sess = tf.Session(config=config)
K.set_session(sess)
from data.datasetMproc import DataSet
import os
import sys
import json

# id map
idmap = {i:i for i in range(1,14)}
cats = open('data/criteo/ids.txt').readlines()
for i,cat in enumerate(cats):
    key = cat.strip()
    idmap[key] = 14 + i

val_data = DataSet('data/val.txt',4584094,idmap,1000,workers=5, queue_size=200)
# load model
mdlPath = sys.argv[1]
mdl = tf.keras.models.load_model(mdlPath)
res = mdl.evaluate(val_data, steps=len(val_data),max_queue_size=10)
print(res)

