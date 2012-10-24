#default configuration file

class config(object):
    pass
cfg=config()

cfg.dbpath="/home/databases/"
cfg.localdata="/esat/nereid/mpederso/VOC07/" #used for checkpoints
cfg.db="VOC"
cfg.year="2007"
cfg.cls="bicycle"
#cfg.fy=8#remember small
#cfg.fx=3
#cfg.lev=3
cfg.interv=10
#cfg.ovr=0.45
cfg.sbin=8
cfg.maxpos=2000#120
cfg.maxtest=2000#100
cfg.maxtestfull=10000
cfg.maxneg=200#120
cfg.maxexamples=10000
cfg.maxnegfull=2000
#cfg.deform=True
#cfg.usemrf=False
#cfg.usefather=True
#cfg.bottomup=False
#cfg.inclusion=False
#cfg.initr=1
#cfg.ratio=1
#cfg.mpos=0.5
#cfg.posovr=0.7
cfg.hallucinate=1
#cfg.numneginpos=5
#cfg.useflipos=True
#cfg.useflineg=True
cfg.svmc=0.001#0.001#0.002#0.004
cfg.convPos=0.005
cfg.convNeg=1.05
cfg.show=False
#cfg.thr=-2
cfg.multipr=4
cfg.negit=10#10
cfg.posit=8
cfg.perc=0.15
cfg.comment="I shuld get more than 84... hopefully"
#cfg.numneg=0#not used but necessary
cfg.testname="./data/test"
cfg.savefeat=False #save precomputed features 
cfg.loadfeat=True
#cfg.useprior=False
#cfg.ovr=0.5
cfg.k=1.0
cfg.small=False
#cfg.dense=0
#cfg.ovrasp=0.3
#cfg.minnegincl=0
#cfg.sortneg=False
#cfg.bestovr=False
cfg.useRL=False
cfg.kmeans=False
cfg.resize=None
#cfg.ranktr=500
#cfg.occl=False
#cfg.small2=False
#cfg.saveVOC=False
cfg.auxdir=""
#cfg.denseinit=False
cfg.checkpoint=False #use checkpoints to recover partial trainings
cfg.forcescratch=False #force to start from scratch even though there are checkpoints
#cfg.variablecache=False
cfg.rotate=False
cfg.valreg=0.01
cfg.initdef=0.01
cfg.rescale=True
cfg.useclip=False
cfg.posthr=-2 #if score is smaller than this value detection is discarded
cfg.maxHOG=50
cfg.neginpos=False
cfg.numneg= 10 #number of hard negatives to collect per image
cfg.N=2 #size of a part
cfg.localshow=False
cfg.trunc=0
cfg.lb=0.001
cfg.useSGD=True

#trade-off speed accuracy
#now in training when not using bb it is using force parts
cfg.usebbPOS=False #note for bb numhyp is the total number of detections while for the normal method is the number of hypotheses per scale
cfg.numhypPOS=5
cfg.aiterPOS=3
cfg.restartPOS=0
cfg.intervPOS=10

cfg.usebbNEG=False
cfg.numhypNEG=1
cfg.aiterNEG=3
cfg.restartNEG=0
cfg.intervNEG=10

cfg.usebbTEST=False
cfg.numhypTEST=1
cfg.aiterTEST=3
cfg.restartTEST=0
cfg.intervTEST=10

cfg.useswTEST=False
cfg.swstepy=-1
cfg.swstepx=-1

cfg.userot=False
cfg.rotangle=20

#cfg.savedir=InriaPosData(basepath=dbpath).getStorageDir() #where to save
#    mydebug=False
#    if mydebug:
#        cfg.multipr=False
#        cfg.maxpos=10
#        cfg.maxneg=10
#        cfg.maxtest=10
#        cfg.maxexamples=1000
import subprocess
cfg.version=subprocess.check_output(["git","log","--pretty=oneline","--abbrev-commit","-1"])


