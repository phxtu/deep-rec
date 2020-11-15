import tensorflow as tf
import tensorflow.keras.backend as K
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.36
sess = tf.Session(config=config)
K.set_session(sess)
from data.datasetMproc import DataSet
from models.deepFm import DeepFm
from models.xdeepFm import XDeepFm
import os
import sys
import json,time

mdls = ['deepFm', 'xdeepFm']
mdlName = sys.argv[1]
assert mdlName in mdls, "%s not supported now, must be one of %s"%(mdlName, mdls)
outdir='mdls/%s/%s'%(mdlName, time.strftime("%Y%m%d_%H%M%S", time.localtime()))
if not os.path.exists(outdir):
    os.makedirs(outdir)

# config
conf = {
    'batch':128,
    'lr':1e-3,
    'epochs':300,
    'emb_size': 10,
    'mlps':[400, 400, 400],
    'cins':[200]*3,
    'useFm':True,
    'useDeep':True,
    'dropFm':[0,0],
    'dropDeeps':[0]*4,
    'seed':666,
    'logdir':'%s/logs'%outdir,
    'mdlfig':'%s/mdl.png'%outdir,
}
if len(sys.argv) > 2:
    for kv in sys.argv[2:]:
        k, v = kv.split('=')
        if k in ['batch', 'epochs', 'emb_size', 'seed']:
            conf[k] = int(v)
        elif k in ['lr']:
            conf[k] = float(v)
        elif k in ['mlps', 'cins']:
            conf[k] = [int(e) for e in v.split(',')]
        elif k in ['dropFm','dropDeeps']:
            conf[k] = [float(e) for e in v.split(',')]
        elif k in ['useFm','useDeep']:
            conf[k] = False if v == '0' else True
json.dump(conf,open("%s/conf.json"%outdir,'w'),indent=4)
# id map
idmap = {i:i for i in range(1,14)}
cats = open('data/criteo/ids.txt').readlines()
for i,cat in enumerate(cats):
    key = cat.strip()
    idmap[key] = 14 + i
# training & val data 4584094
trn_data = DataSet('data/train.txt',1e6,idmap,conf['batch'],workers=5, queue_size=200)
val_data = DataSet('data/val.txt',1e6,idmap,conf['batch'],workers=5, queue_size=200)
# build model
if mdlName == 'deepFm':
    Model = DeepFm(1+len(idmap),conf['emb_size'],conf['mlps'],conf['useFm'],conf['useDeep'],conf['dropFm'], conf['dropDeeps'], conf['seed'])
elif mdlName == 'xdeepFm':
    Model = XDeepFm(1+len(idmap),conf['emb_size'],conf['mlps'],conf['cins'],conf['useFm'],conf['useDeep'],conf['dropFm'], conf['dropDeeps'], conf['seed'])
mdl = Model.build()
mdl.summary()
tf.keras.utils.plot_model(mdl, conf['mdlfig'], show_shapes=True)
# compile model
mdl.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=conf['lr']),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['AUC'],
)
# fit
mdl.fit(
    trn_data,epochs=conf['epochs'],validation_data=val_data,
    steps_per_epoch=len(trn_data),validation_steps=len(val_data),
    max_queue_size=50,workers=1,use_multiprocessing=False,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=60,restore_best_weights=False),
        #tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
        tf.keras.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filepath='%s/model.h5'%outdir,
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir=conf['logdir']),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",factor=0.2,patience=20,verbose=1,
            mode="auto",min_delta=0.0001,cooldown=0,min_lr=1e-6
        )
   ]
)
