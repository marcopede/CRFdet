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
from extra import locatePoints,locatePointsInter

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
    
    #anchor=[6,4.5, 6,7.5, 12.5,6.5, 12.5,11, 13.75,9, 11.5,9, 6,11, 6.25,13, 10,7.5, 10,10.75]#points lower resolution
    #anchor=[8,5, 8,10.2, 16.5,8, 16.5,15, 17.5,12, 16,12, 8,14, 8,18, 13,10, 13,14 ]#best results for the moment
    #anchor=numpy.array([7.5,5, 7.5,7.5, 13,6, 13,11.5, 14.5,9, 12,9, 7.5,10, 7.5,13, 11.5,7, 11.5,11 ])#model for nobbox2
    error=numpy.zeros(20)
    totgpi=0
    totgp=0
    for ii,res in enumerate(itr):
        reducebb=True
        if reducebb:
            for idd,det in enumerate(res):
                auxbb=numpy.zeros(4)
                w=det["bbox"][3]-det["bbox"][1]
                h=det["bbox"][2]-det["bbox"][0]
                if det["id"]==0:#left facing
                    auxbb[1]=det["bbox"][1]+0.2*w
                    auxbb[3]=det["bbox"][3]-0.1*w
                    auxbb[0]=det["bbox"][0]+0.2*h
                    auxbb[2]=det["bbox"][2]-0.1*h
                else:
                    auxbb[1]=det["bbox"][1]+0.1*w
                    auxbb[3]=det["bbox"][3]-0.2*w
                    auxbb[0]=det["bbox"][0]+0.2*h
                    auxbb[2]=det["bbox"][2]-0.1*h
                res[idd]["bbox"]=auxbb
        if show:
            im=myimread(arg[ii]["file"])
            if tsImages[ii]["bbox"]!=[]:
                detectCRF.visualize2(res[:3],cfg.N,im,bb=tsImages[ii]["bbox"],line=True)
            else:
                detectCRF.visualize2(res[:3],cfg.N,im)
            print [x["scr"] for x in res[:5]]
            raw_input()
        facial=False
        if facial:    #evaluate facial features position
            anchor=models[res[0]["id"]]["facial"]
            if res[0]["id"]==1:
                inv=[14,15, 12,13, 6,7, 4,5, 8,9, 10,11, 2,3, 0,1, 18,19, 16,17]
                anchor=anchor[inv]
            fp=tsImages[ii]["facial"]
            py,px=tsImages[ii]["bbox"][0][0:2]
            gtfp=fp.copy()
            gtfp[0:-1:2]=fp[1::2]+py
            gtfp[1::2]=fp[0:-1:2]+px
            pylab.plot(gtfp[1::2],gtfp[0:-1:2],"or", markersize=9)
            efp=numpy.array(locatePoints(res[:1],cfg.N,anchor)[0])#,6,7,6,10,
            efpi=numpy.array(locatePointsInter(res[:1],cfg.N,anchor)[0])#,6,7,6,10,
            #print "Estimated",efp
            #print "Ground Truth",py+fp[1],px+fp[0]
            pylab.plot(efp[1::2],efp[0:-1:2],"ob",markersize=7)
            pylab.plot(efpi[1::2],efpi[0:-1:2],"og",markersize=7)
            intocu=numpy.sqrt(((gtfp[13]+gtfp[15])/2.0-(gtfp[1]+gtfp[3])/2.0)**2+((gtfp[12]+gtfp[14])/2.0-(gtfp[0]+gtfp[2])/2.0)**2)
            dist=numpy.sqrt(numpy.sum(numpy.reshape(gtfp-efp,(-1,2))**2,1))
            disti=numpy.sqrt(numpy.sum(numpy.reshape(gtfp-efpi,(-1,2))**2,1))
            #print "Pupil Left",(gtfp[1]+gtfp[3])/2,"Pupil Right",(gtfp[13]+gtfp[15])/2
            print "Inter ocular distance",intocu
            print "Dist Near",dist
            print "Dist Inter",disti
            print "Nearest",numpy.sum(dist)
            print "Inter",numpy.sum(disti)
            gp=numpy.sum(dist<0.1*intocu)
            gpi=numpy.sum(disti<0.1*intocu)
            print "Good points Near",gp
            print "Good points Inter",gpi
            error=error+gtfp-efpi
            pylab.draw()
            pylab.show()
            print "Error",error
            totgpi+=gpi
            totgp+=gp
            print "Global Avrg",float(totgp)/(ii+1)
            print "Global Avrg Inter",float(totgpi)/(ii+1)
            print
            raw_input()
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
    print "AP=",ap
    #save in different formats
    if type(save)==str:
        testname=save
        util.savedetVOC(detVOC,testname+".txt")
        util.save(testname+".det",{"det":ltdet[:500]})#takes a lot of space use only first 500
        util.savemat(testname+".mat",{"tp":tp,"fp":fp,"scr":scr,"tot":tot,"rc":rc,"pr":pr,"ap":ap})
        pylab.savefig(testname+".png")
    return ap


#use a different number of hypotheses
def test(x):
    return detectCRF.test(x,show=False,inclusion=False,onlybest=False) #in bicycles is 

def testINC(x):
    return detectCRF.test(x,show=False,inclusion=True,onlybest=True) #in bicycles is better and faster with 1 hypotheses

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
        #cfg.maxpos = 50
        #cfg.maxneg = 20
        #cfg.maxexamples = 10000
    else: #or set here the parameters
        print "Loading defautl configuration config.py"
        from config import * #default configuration      
    cfg.cls=sys.argv[1]
    cfg.numcl=1
    #cfg.dbpath="/home/owner/databases/"
    cfg.dbpath="/users/visics/mpederso/databases/"
    cfg.testpath="./data/test/"#"./data/CRF/12_09_19/"
    cfg.testspec="force-bb"#"full2"
    cfg.db="AFW"
    #cfg.db="imagenet"
    #cfg.cls="tandem"
    #cfg.N=
        

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
            tsImages=numpy.concatenate((tsPosImages,tsNegImages),0)
            tsImagesFull=getRecord(VOC07Data(select="all",cl="%s_test.txt"%cfg.cls,
                            basepath=cfg.dbpath,
                            usetr=True,usedf=False),10000)
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
    elif cfg.db=="imagenet":
        tsPosImages=getRecord(imageNet(select="all",cl="%s_test.txt"%cfg.cls,
                        basepath=cfg.dbpath,
                        trainfile="/tandem/",
                        imagepath="/tandem/images/",
                        annpath="/tandem/Annotation/n02835271/",
                        usetr=True,usedf=False),cfg.maxtest)
        tsImages=tsPosImages#numpy.concatenate((tsPosImages,tsNegImages),0)
        tsImagesFull=tsPosImages
    elif cfg.db=="LFW":
        tfold=0 #test fold 0 other 9 for training
        aux=getRecord(LFW(basepath=cfg.dbpath,fold=0),cfg.maxpos)
        trPosImages=numpy.array([],dtype=aux.dtype)
        for l in range(10):
            aux=getRecord(LFW(basepath=cfg.dbpath,fold=l),cfg.maxpos)
            if l==tfold:
                tsImages=getRecord(LFW(basepath=cfg.dbpath,fold=l),cfg.maxtest,facial=True)
            else:
                trPosImages=numpy.concatenate((trPosImages,aux),0)
        trPosImagesNoTrunc=trPosImages
        trNegImages=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxneg)#check if it works better like this
        trNegImagesFull=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxnegfull)
        #test
        #tsImages=getRecord(InriaTestFullData(basepath=cfg.dbpath),cfg.maxtest)
        tsImagesFull=tsImages
    elif cfg.db=="AFW":
        tsImages=getRecord(AFW(basepath=cfg.dbpath),cfg.maxpos)
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
    #it=2;testname="./data/bicycle2_test"

    if 0: #standard configuration
        cfg.usebbTEST=False
        cfg.numhypTEST=1
        cfg.aiterTEST=3
        cfg.restartTEST=0
        cfg.intervTEST=10

    if 0: #used during training
        cfg.usebbTEST=True
        cfg.numhypTEST=50
        cfg.aiterTEST=1
        cfg.restartTEST=0
        cfg.intervTEST=5

    if 1: #personalized
        cfg.usebbTEST=True
        cfg.numhypTEST=50
        cfg.aiterTEST=3
        cfg.restartTEST=0
        cfg.intervTEST=5

    cfg.numcl=2
    cfg.N=4
    cfg.useclip=True
    cfg.useFastDP=True
    #testname="./data/CRF/12_10_02_parts_full/bicycle2_testN2_final"
    #testname="./data/person1_testN2best0"#inria1_inria3"bicycle2_testN4aiter3_final
    #testname="./data/bicycle2_testN4aiter3_final"
    #testname="./data/bicycle2_testN4aiter38"
    #testname="./data/bicycle2_testN36"
    #testname="./data/resultsN2/bicycle2_N2C2_final"
    #testname="./data/afterCVPR/bicycle2_force-bb_final"
    #testname="../../CRFdet/data/afterCVPR/12_01_10/cat2_force-bb_final"
    #testname="data/condor2/person3_full_condor219"
    #testname="data/test/face1_lfw_highres_final"
    #testname="data/test/face1_lfw_sbin44"
    #testname="data/test/face1_lfw_sbin4_high3"
    #testname="data/lfw/face1_nobbox2_final"
    #testname="data/test/face1_facial9_final"
    #testname="data/test2/face1_interp32"
    #testname="data/test2/face1_nopoints_final"
    #testname="data/test2/face2_2mixt_final"
    #testname="data/test2/face2_pose_final"
    testname="data/full/face2_pose_full9"
    #testname="/users/visics/mpederso/code/git/bigger/CRFdet/data/test/face1_lfw_highres_final"
    cfg.trunc=1
    models=util.load("%s.model"%(testname))
    #cfg.numhypTEST=100
    #del models[0]
    cfg.numcl=2
    cfg.E=1
    #cfg.N=2
    #cfg.sbin=4
    cfg.N=models[0]["N"]
    cfg.hallucinate=0
    #del models[0]["thr"]
    #del models[1]["thr"]
    #del models[2]["thr"]
    #del models[3]["thr"]
    #models=util.load("%s%d.model"%(testname,it))
    #just for the new
    #for idm,m in enumerate(models):
    #    models[idm]["cost"]=models[idm]["cost"]*0.2
#        newc=numpy.zeros((8,aux.shape[1],aux.shape[2]),dtype=aux.dtype)
#        newc[:4]=aux
#        models[idm]["cost"]=newc
    ##############test
    #import itertools
    #runtest(models,tsImages,cfg,parallel=False,numcore=4,detfun=lambda x :detectCRF.test(x,numhyp=1,show=False),show=True)#,save="%s%d"%(testname,it))
    runtest(models,tsImagesFull,cfg,parallel=True,numcore=8,show=True,detfun=testINC,save="./AWFpose")

