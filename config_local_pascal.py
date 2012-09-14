#local configuration file example
#rename it config_local.py and set your local configuration

import config

cfg=config.cfg

cfg.multipr=8
cfg.savefeat=False
cfg.loadfeat=False
cfg.savedir=""#"/state/partition1/marcopede/INRIA/hog/"

#cfg.dbpath = "/share/ISE/marcopede/database/" #database path
cfg.dbpath = "/home/owner/databases/" #database path
#cfg.dbpath = "/users/visics/mpederso/databases/" #database path
cfg.maxpos=10000 #maximum number of positive images should be 10000
cfg.maxtest=20000 #0 #maximum number of test images
cfg.maxneg=240 #maximum number of negative images
cfg.maxexamples=20000 #maximum number of examples

if 0:
    cfg.maxpos=32 #maximum number of positive images
    cfg.maxtest=8 #0 #maximum number of test images
    cfg.maxneg=8 #maximum number of negative images
    cfg.maxexamples=20000 #maximum number of examples

cfg.CRF=True
cfg.deform=False
cfg.usemrf=True
cfg.usefather=True
cfg.bottomup=False
cfg.fastBU=False #for the moment only in test
cfg.initr=1
cfg.ratio=1


cfg.db="buffy"#"VOC"#"adondemand"#"VOC"#"inria"#"ivan"#"VOC"
#additional for pascal
cfg.lev=[3,3,3,3,3]
cfg.numcl=3
cfg.numneginpos=0#6/cfg.numcl #reduce the number of negatives
cfg.svmc=0.001#0.001
cfg.cls="person"
cfg.year="2007"
cfg.show=True
cfg.thr=-2      #threshold positives
cfg.mythr=-10   #threshold cascade
cfg.posovr=0.75#0.65
cfg.perc=0.12
cfg.posit=15#8
cfg.negit=15#5
cfg.useprior=False
cfg.dense=0
cfg.k=1.0#3:CRF#0.3:CF
cfg.useRL=True
cfg.kmeans=True
cfg.bestovr=True
cfg.resize=1.0
cfg.small=False
cfg.occl=False
cfg.usebow=False
cfg.trunc=0
cfg.noiselev = 0.05
cfg.ranktr=100 #for CRF should be 500
cfg.denseinit=False
cfg.checkpoint=True #restart form the last point
cfg.variablecache=False #use a cache that varies depending on the number of positive examples and uses maxexamples as muximum value
cfg.rotate=True

cfg.small2=False
cfg.hallucinate=1 #important to set to 2 otherwise small2 would not work

cfg.comment    =""
#cfg.testname="./data/11_02_28/iria_full" #location of the current test
#cfg.testpath="./data/PASCAL/12_02_16_buffer/"
#cfg.testpath="./data/debug/"
cfg.testpath="./data/CRF/12_09_15/"
cfg.testspec="buffy"
#cfg.testname="./data/11_03_10/%s_%d_3levnompos_hres"%(cfg.cls,cfg.numcl)

boost=False
if boost:
    cfg.lev=[3]*10
    cfg.deform=True
    cfg.CRF=False
    cfg.svmc=0.001
    cfg.lev=[3]
    cfg.maxpos=2000 #maximum number of positive images
    cfg.maxtest=2000 #0 #maximum number of test images
    cfg.maxneg=240 #maximum number of negative images
    cfg.maxexamples=20000 #maximum number of examples
    cfg.posovr=0.75
    cfg.ranktr=10
    cfg.useRL=True
    cfg.kmeans=True
    cfg.testpath="./data/BOOST/12_07_01/"
    cfg.testspec="rank10"
    cfg.small2=False
    cfg.hallucinate=1

if cfg.db=="adondemand":
    cfg.deform=False
    cfg.CRF=False
    cfg.svmc=0.01
    cfg.lev=[3]
    cfg.maxpos=1 #maximum number of positive images
    cfg.maxtest=100 #0 #maximum number of test images
    cfg.maxneg=20 #maximum number of negative images
    cfg.maxexamples=2000 #maximum number of examples
    cfg.cls="escote"
    cfg.posovr=0.5
    cfg.ranktr=2000
    cfg.numcl=1
    cfg.useRL=True
    cfg.kmeans=False
    cfg.testpath="./data/ADONDEMAND/12_06_29/"
    cfg.testspec="video"
    cfg.small2=True
    cfg.hallucinate=2
    cfg.saveVOC=True

#cfg.cls="inria"
if cfg.db=="inria":
    cfg.cls="person"
    cfg.testspec="inria-crf"
    cfg.multipr=8
    #cfg.usefather=False
    #cfg.fastBU=True
    cfg.sortneg=True
    #cfg.show=False
    cfg.useRL=True
    cfg.negit=10
    cfg.posit=10
    cfg.testpath="./data/CRFRot/12_08_29/"#"./data/INRIA/12_02_20/"
    cfg.minnegincl=0.5
    cfg.maxpos=4000 #maximum number of positive images
    cfg.maxtest=1000 #maximum number of test images
    cfg.maxneg=240 #maximum number of negative images
    cfg.maxexamples=30000 #maximum number of examples
    cfg.lev=[3,3]#[3,3]
    cfg.numcl=2
    cfg.numneginpos=1#10
    cfg.k=1.0#3#0.3
    cfg.small=False#1
    cfg.dense=0
    cfg.posovr=0.75
    cfg.perc=0.12 #how many examples to discard
    cfg.mpos=0.5
    cfg.resize=1.0
    cfg.kmeans=True
    cfg.occl=False#True
    cfg.ratio=1
    cfg.oldlearning=False
    cfg.usebuffer=False#True
    cfg.lenbuf=240
    cfg.usebufpos=False
    cfg.usebufneg=True
    cfg.usebuftest=False

if cfg.db=="cvc02":
    cfg.cls="person"
    cfg.testspec="full100"
    cfg.deform=True
    cfg.sbin=8
    cfg.useRL=False
    cfg.testpath="./data/CVC02/11_07_25/"
    cfg.sortneg=True
    cfg.show=True
    cfg.posit=10
    cfg.negit=10
    cfg.minnegincl=0.5
    cfg.maxpos=2032 #maximum number of positive images
    cfg.maxneg=2000 #maximum number of negative images
    cfg.maxtest=2000 #maximum number of test images
    cfg.maxexamples=10000 #maximum number of examples
    cfg.lev=[3,3]
    cfg.numcl=1
    cfg.numneginpos=0
    cfg.k=0.3
    cfg.small=True
    cfg.dense=0
    cfg.posovr=0.75
    cfg.perc=0.2
    cfg.mpos=0.5
    cfg.kmeans=False
    cfg.occl=True

if cfg.db=="ivan":
    cfg.cls="person"
    cfg.testspec="test"
    cfg.multipr=4
    cfg.deform=True
    cfg.sbin=8
    cfg.useRL=True
    cfg.testpath="./data/IVAN/11_10_03/"
    cfg.sortneg=True
    cfg.show=True
    cfg.posit=10
    cfg.negit=10
    #cfg.minnegincl=0.5
    cfg.minnegincl=0
    cfg.maxpos=1000 #maximum number of positive images
    cfg.maxneg=400 #maximum number of negative images
    cfg.maxtest=250 #maximum number of test images
    cfg.maxexamples=20000 #maximum number of examples
    cfg.lev=[2,2,2,2]
    cfg.numcl=3
    cfg.numneginpos=0#10
    cfg.k=0.3
    cfg.small=True
    cfg.dense=0
    cfg.posovr=0.7
    cfg.perc=0.2
    cfg.mpos=0.5
    cfg.kmeans=False
    cfg.occl=True
#### for clustering    
    cluster=True
    if cluster:
        cfg.posit=3
        cfg.negit=10
        cfg.occl=False
        cfg.small=False
        cfg.posovr=0.4
        cfg.numcl=1

if cfg.db=="ransac":
    #cfg.db="VOC"
    #cfg.cls="bicycle"
    cfg.multipr=4
    cfg.testspec="full100"
    cfg.deform=True
    cfg.sbin=8
    cfg.useRL=False
    cfg.testpath="./data/ransac/11_11_25/"
    cfg.sortneg=True
    cfg.useRL=True
    cfg.show=True
    cfg.posit=5
    cfg.negit=10
    cfg.minnegincl=0.5
    cfg.maxpos=1 #maximum number of positive images
    cfg.maxneg=20 #maximum number of negative images
    cfg.maxtest=20 #maximum number of test images
    cfg.maxexamples=10000 #maximum number of examples
    cfg.lev=[3,3]
    cfg.numcl=1
    cfg.numneginpos=0
    cfg.k=0.3
    cfg.small=True
    cfg.dense=0
    cfg.posovr=0.7
    cfg.perc=0.2
    cfg.mpos=0.5
    cfg.kmeans=False
    cfg.occl=True

cfg.debug = False#False #debug mode

if cfg.debug:
    #cfg.posovr=0.5#0.65
    #cfg.useprior=False
    #cfg.lev=[3,3]
    #cfg.testspec="Debug"
    cfg.show=True
    cfg.posit=8
    cfg.negit=5
    #cfg.testspec="debug"
    cfg.multipr=4
    cfg.maxpos=20#120
    cfg.maxtest=20#100
    cfg.maxneg=20#120
    cfg.description="False"
    #cfg.testname=cfg.testname+"_debug"

