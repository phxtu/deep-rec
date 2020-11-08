#coding:utf-8
import numpy as np
import tensorflow as tf
from multiprocessing import Manager,Pool
from random import shuffle, randint
from .dataProcess import process

class DataSet(tf.keras.utils.Sequence):
    def __init__(self, fin, lines, idmap, batch=32, workers=3, queue_size=20, sr=-1):
        self.files = [ln.strip() for ln in open(fin).readlines()]
        self.lines = lines
        self.idmap = idmap
        self.batch = batch
        self.workers = min(workers, len(self.files))
        self.que = Manager().Queue(queue_size)
        self.pool = Pool(workers)
        self.sr = sr
        print('ready to start workers')
        self.start()
        print('%d workers have started'%self.workers)

    def __len__(self):
        return int(np.ceil(self.lines / self.batch))

    def start(self):
        shuffle(self.files)
        nfiles = len(self.files)//self.workers
        for i in range(self.workers):
            worker = Worker(self.files[i*nfiles:(i+1)*nfiles], self.idmap, self.batch, sr=self.sr)
            self.pool.apply_async(worker.put, (self.que,))
            print('start worker %d'%i)
        self.pool.close()

    def __getitem__(self, idx):
        return self.que.get(True)

class Worker(object):
    def __init__(self, files, idmap, batch=32, sr=-1):
        self.files = files
        self.idmap = idmap
        self.finx = 0
        self.fr = None #open(self.files[self.finx])
        self.batch = batch
        self.sr = sr
    
    def parseLine(self, parts):
        y = int(parts[0]) if parts[0] else 0
        featI, featV = process(parts, self.idmap)
        return y, featI, featV

    def getBatch(self):
        if self.fr is None: self.fr = open(self.files[self.finx])
        y, featI, featV = [0]*self.batch, [0]*self.batch, [0]*self.batch
        cnt = 0
        while cnt < self.batch:
            try:
                ln = next(self.fr)
            except StopIteration as e:
                self.fr.close()
                self.finx += 1
                if self.finx >= len(self.files):
                    self.finx = 0
                self.fr = open(self.files[self.finx])
                continue
            ln = ln.rstrip('\n').split('\t')
            if not ln or (ln[0] == '0' and self.sr>0 and randint(1,100)>self.sr): continue
            try:
                y[cnt],featI[cnt],featV[cnt] = self.parseLine(ln)
            except: continue
            cnt += 1
        return ({'featI':np.array(featI,dtype=int),'featV':np.array(featV)}, np.array(y,dtype=int))

    def put(self, que):
        while True:
            que.put(self.getBatch(),True)


if __name__ == "__main__":
    import sys
    fin = sys.argv[1]
    dataset = DataSet(fin, 6, 2, workers=1)
    for x,y in dataset:
        print(x,y)
    
