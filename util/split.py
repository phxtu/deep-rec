#coding:utf-8
from random import randint

def split(fin, fout, nparts=100):
    fouts = [open(fout+"_%d"%i, 'w') for i in range(nparts)]
    with open(fin) as fi:
        for ln in fi:
            inx = randint(0, nparts-1)
            fo = fouts[inx]
            fo.write(ln)

if __name__ == "__main__":
    import sys
    fin, fout, nparts = sys.argv[1:4]
    split(fin, fout, int(nparts))
