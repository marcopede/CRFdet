import matplotlib
matplotlib.use("Agg") #if run outside ipython do not show any figure
from database import *
from multiprocessing import Pool
import util
import pyrHOG2
#import pyrHOG2RL
import extra
import VOCpr
import model
import time
import copy
import itertools
import sys

if __name__ == '__main__':

    if 0: #use the configuration file
        print "Loading defautl configuration config.py"
        from config import * #default configuration      

        if len(sys.argv)>2: #specific configuration
            print "Loading configuration from %s"%sys.argv[2]
            import_name=sys.argv[2]
            exec "from config_%s import *"%import_name
            
        #cfg.cls=sys.argv[1]
        cfg.useRL=False#for the moment
        cfg.show=False
        cfg.auxdir=""
        cfg.numhyp=5
        cfg.rescale=True
        cfg.numneg= 10
        bias=100
        cfg.bias=bias
        #just for a fast test
        #cfg.maxpos = 50
        #cfg.maxneg = 20
        #cfg.maxexamples = 10000
    else: #or set here the parameters
        print "Loading defautl configuration config.py"
        from config import * #default configuration      
        #cfg.cls=sys.argv[1]
        cfg.numcl=3
        #cfg.dbpath="/home/owner/databases/"
        cfg.dbpath="/users/visics/mpederso/databases/"
        cfg.testpath="./data/"#"./data/CRF/12_09_19/"
        cfg.testspec="right"#"full2"
        cfg.db="VOC"
        #cfg.N=
       

    import pylab as pl
    import util
    import detectCRF
    #det=util.load("./data/CRF/12_10_02_parts_full/bicycle2_testN1_final.det")["det"]
    det=util.load("./data/CRF/12_10_02_parts_full/bicycle2_testN2_final.det")["det"]
    imgpath=cfg.dbpath+"VOC2007/VOCdevkit/VOC2007/JPEGImages/"
    for idl,l in enumerate(det):
        img=util.myimread(imgpath+l["idim"])
        #pl.figure(100)        
        #pl.clf()
        #pl.imshow(img)
        detectCRF.visualize2([l],2,img,text="rank:%d"%(idl))
        raw_input()







