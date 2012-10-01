# training of the new CRF model
# denseCRF [category] [configuration]

##################some import
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
import detectCRF

def runtest(models,tsImages,cfg,parallel=True,numcore=4,detfun=detectCRF.test,save=False,show=False,pool=None):

    #parallel=True
    #cfg.show=not(parallel)
    #numcore=4
    #mycfg=
    if parallel:
        if pool!=None:
            mypool=pool #use already created pool
        else:
            mypool = Pool(numcore)
    arg=[]

    for idl,l in enumerate(tsImages):
        #bb=l["bbox"]
        #for idb,b in enumerate(bb):
        arg.append({"idim":idl,"file":l["name"],"idbb":0,"bbox":[],"models":models,"cfg":cfg,"flip":False})    

    print "----------Test-----------"
    ltdet=[];
    if not(parallel):
        #itr=itertools.imap(detectCRF.test,arg)        
        #itr=itertools.imap(lambda x:detectCRF.test(x,numhyp=1),arg) #this can also be used       
        itr=itertools.imap(detfun,arg)
    else:
        #itr=mypool.map(detectCRF.test,arg)
        itr=mypool.imap(detfun,arg) #for parallle lambda does not work

    for ii,res in enumerate(itr):
        if show:
            im=myimread(arg[ii]["file"])
            detectCRF.visualize2(res[:3],im)
            print [x["scr"] for x in res[:5]]
        ltdet+=res

    if parallel:
        if pool==None:
            mypool.close() 
            mypool.join() 

    #sort detections
    ltosort=[-x["scr"] for x in ltdet]
    lord=numpy.argsort(ltosort)
    aux=[]
    for l in lord:
        aux.append(ltdet[l])
    ltdet=aux

    #save on a file and evaluate with annotations
    detVOC=[]
    for l in ltdet:
        detVOC.append([l["idim"].split("/")[-1].split(".")[0],l["scr"],l["bbox"][1],l["bbox"][0],l["bbox"][3],l["bbox"][2]])

    #plot AP
    tp,fp,scr,tot=VOCpr.VOCprRecord(tsImages,detVOC,show=False,ovr=0.5)
    pylab.figure(15,figsize=(4,4))
    pylab.clf()
    rc,pr,ap=VOCpr.drawPrfast(tp,fp,tot)
    pylab.draw()
    pylab.show()
    #save in different formats
    if type(save)==str:
        testname=save
        util.savedetVOC(detVOC,testname+".txt")
        #util.save(testname+".det",{"det":ltdet})#takes a lot of space
        util.savemat(testname+".mat",{"tp":tp,"fp":fp,"scr":scr,"tot":tot,"rc":rc,"pr":pr,"ap":ap})
        pylab.savefig(testname+".png")


#use a different number of hypotheses
def test1hypINC(x):
    return detectCRF.test(x,show=False,inclusion=True,onlybest=True) #in bicycles is 

def test3hypINC(x):
    return detectCRF.test(x,show=False,inclusion=True,onlybest=True) #in bicycles is better and faster with 1 hypotheses

def test50hypINCBB(x):
    return detectCRF.test(x,show=False,inclusion=True,onlybest=True) #in bicycles is 

#use a different number of hypotheses
def test1hyp(x):
    return detectCRF.test(x,show=False,inclusion=False,onlybest=False) #in bicycles is better and faster with 1 hypotheses


########################## load configuration parametes
if __name__ == '__main__':

    if 0: #use the configuration file
        print "Loading defautl configuration config.py"
        from config import * #default configuration      

        if len(sys.argv)>2: #specific configuration
            print "Loading configuration from %s"%sys.argv[2]
            import_name=sys.argv[2]
            exec "from config_%s import *"%import_name
            
        cfg.cls=sys.argv[1]
        cfg.useRL=False#for the moment
        cfg.show=False
        cfg.auxdir=""
        cfg.numhyp=5
        cfg.rescale=True
        cfg.numneg= 10
        bias=100
        cfg.bias=bias
        #just for a fast test
        cfg.maxpos = 50
        cfg.maxneg = 20
        cfg.maxexamples = 10000
    else: #or set here the parameters
        print "Loading defautl configuration config.py"
        from config import * #default configuration      
        cfg.cls=sys.argv[1]
        cfg.numcl=3
        #cfg.dbpath="/home/owner/databases/"
        cfg.dbpath="/users/visics/mpederso/databases/"
        cfg.testpath="./data/"#"./data/CRF/12_09_19/"
        cfg.testspec="right"#"full2"
        cfg.db="inria"
        

    testname=cfg.testpath+cfg.cls+("%d"%cfg.numcl)+"_"+cfg.testspec
    ########################load training and test samples
    if cfg.db=="VOC":
        if cfg.year=="2007":
            #test
            tsPosImages=getRecord(VOC07Data(select="pos",cl="%s_test.txt"%cfg.cls,
                            basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                            usetr=True,usedf=False),cfg.maxtest)
            tsNegImages=getRecord(VOC07Data(select="neg",cl="%s_test.txt"%cfg.cls,
                            basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                            usetr=True,usedf=False),cfg.maxneg)
            #tsImages=numpy.concatenate((tsPosImages,tsNegImages),0)
            tsImages=numpy.concatenate((tsPosImages,tsNegImages[:0]),0)
            tsImagesFull=getRecord(VOC07Data(select="all",cl="%s_test.txt"%cfg.cls,
                            basepath=cfg.dbpath,
                            usetr=True,usedf=False),5000)
    elif cfg.db=="buffy":
        trPosImages=getRecord(Buffy(select="all",cl="trainval.txt",
                        basepath=cfg.dbpath,
                        trainfile="buffy/",
                        imagepath="buffy/images/",
                        annpath="buffy/",
                        usetr=True,usedf=False),cfg.maxpos)
        trPosImagesNoTrunc=trPosImages
        trNegImages=getRecord(DirImages(imagepath=cfg.dbpath+"INRIAPerson/train_64x128_H96/neg/"),cfg.maxneg)
        trNegImagesFull=trNegImages
        #test
        tsPosImages=getRecord(Buffy(select="all",cl="test.txt",
                        basepath=cfg.dbpath,
                        trainfile="buffy/",
                        imagepath="buffy/images/",
                        annpath="buffy/",
                        usetr=True,usedf=False),cfg.maxtest)
        tsImages=tsPosImages#numpy.concatenate((tsPosImages,tsNegImages),0)
        tsImagesFull=tsPosImages
    elif cfg.db=="inria":
        trPosImages=getRecord(InriaPosData(basepath=cfg.dbpath),cfg.maxpos)
        trPosImagesNoTrunc=trPosImages
        trNegImages=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxneg)#check if it works better like this
        trNegImagesFull=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxnegfull)
        #test
        tsImages=getRecord(InriaTestFullData(basepath=cfg.dbpath),cfg.maxtest)
        tsImagesFull=tsImages

    ##############load model
    for l in range(cfg.posit):
        try:
            models=util.load("%s%d.model"%(testname,l))
            print "Loading Model %d"%l
        except:
            break
    #it=l-1
    #models=util.load("%s%d.model"%(testname,it))
    ######to comment down
    #it=6;testname="./data/person3_right"
    #it=12;testname="./data/CRF/12_09_23/bicycle3_fixed"
    #it=4;testname="./data/bicycle3_full"
    it=0;testname="./data/inria1_inria"
    models=util.load("%s%d.model"%(testname,it))
    #just for the new
#    for idm,m in enumerate(models):
#        aux=models[idm]["cost"]
#        newc=numpy.zeros((8,aux.shape[1],aux.shape[2]),dtype=aux.dtype)
#        newc[:4]=aux
#        models[idm]["cost"]=newc
    ##############test
    #import itertools
    #runtest(models,tsImages,cfg,parallel=False,numcore=4,detfun=lambda x :detectCRF.test(x,numhyp=1,show=False),show=True)#,save="%s%d"%(testname,it))
    runtest(models,tsImages,cfg,parallel=True,numcore=4,show=True,detfun=test50hypINCBB,save="./bicycleFullBB50Clip")

