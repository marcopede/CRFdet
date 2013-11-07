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
    error=numpy.zeros(12)
    totgpi=0
    totgp=0
    project=False
    if project:
        for idl,l in enumerate(models[:cfg.numcl]):
            pylab.figure(100+idl)
            import drawHOG
            import model
            im = drawHOG.drawHOG(model.convert2(l["ww"][0],cfg.N,cfg.E))
            pylab.imshow(im)
    for ii,res in enumerate(itr):
        reducebb=True
        if reducebb:
            for idd,det in enumerate(res):
                auxbb=numpy.zeros(4)
                w=det["bbox"][3]-det["bbox"][1]
                h=det["bbox"][2]-det["bbox"][0]
                #for AWF
                #if det["id"]==0:#left facing
                #    auxbb[1]=det["bbox"][1]+0.2*w
                #    auxbb[3]=det["bbox"][3]-0.1*w
                #    auxbb[0]=det["bbox"][0]+0.2*h
                #    auxbb[2]=det["bbox"][2]-0.1*h
                #else:
                #    auxbb[1]=det["bbox"][1]+0.1*w
                #    auxbb[3]=det["bbox"][3]-0.2*w
                #    auxbb[0]=det["bbox"][0]+0.2*h
                #    auxbb[2]=det["bbox"][2]-0.1*h
                #for aflw
                #frontal
                if cfg.numcl==2:
                    if det["id"]==0:#left facing
                        auxbb[1]=det["bbox"][1]+0.2*w
                        auxbb[3]=det["bbox"][3]-0.1*w
                        auxbb[0]=det["bbox"][0]+0.2*h
                        auxbb[2]=det["bbox"][2]-0.1*h
                    elif det["id"]==2:#right facing
                        auxbb[1]=det["bbox"][1]+0.1*w
                        auxbb[3]=det["bbox"][3]-0.2*w
                        auxbb[0]=det["bbox"][0]+0.1*h
                        auxbb[2]=det["bbox"][2]-0.1*h
                    #lateral
                    elif det["id"]==1:#left facing
                        auxbb[1]=det["bbox"][1]+0.2*w
                        auxbb[3]=det["bbox"][3]#-0.2*w
                        auxbb[0]=det["bbox"][0]+0.2*h
                        auxbb[2]=det["bbox"][2]-0.1*h
                    elif det["id"]==3:#right facing
                        auxbb[1]=det["bbox"][1]#+0.1*w
                        auxbb[3]=det["bbox"][3]-0.2*w
                        auxbb[0]=det["bbox"][0]+0.2*h
                        auxbb[2]=det["bbox"][2]-0.1*h
                if cfg.numcl==4:
                    #todo
                    #auxbb[1]=det["bbox"][1]+0.1*w
                    #auxbb[3]=det["bbox"][3]-0.1*w
                    #auxbb[0]=det["bbox"][0]+0.1*h
                    #auxbb[2]=det["bbox"][2]-0.1*h
                    if det["id"]==0:#left facing
                        auxbb[1]=det["bbox"][1]+0.1*w
                        auxbb[3]=det["bbox"][3]-0.1*w
                        auxbb[0]=det["bbox"][0]+0.1*h
                        auxbb[2]=det["bbox"][2]-0.1*h
                    elif det["id"]==1:#right facing
                        auxbb[1]=det["bbox"][1]+0.1*w
                        auxbb[3]=det["bbox"][3]-0.1*w
                        auxbb[0]=det["bbox"][0]+0.1*h
                        auxbb[2]=det["bbox"][2]-0.1*h
                    elif det["id"]==2:#left facing
                        auxbb[1]=det["bbox"][1]+0.2*w
                        auxbb[3]=det["bbox"][3]#-0.1*w
                        auxbb[0]=det["bbox"][0]+0.1*h
                        auxbb[2]=det["bbox"][2]-0.1*h
                    elif det["id"]==3:#right facing
                        auxbb[1]=det["bbox"][1]#+0.1*w
                        auxbb[3]=det["bbox"][3]-0.2*w
                        auxbb[0]=det["bbox"][0]+0.2*h
                        auxbb[2]=det["bbox"][2]-0.1*h
                    elif det["id"]==4:#left facing
                        auxbb[1]=det["bbox"][1]+0.1*w
                        auxbb[3]=det["bbox"][3]-0.1*w
                        auxbb[0]=det["bbox"][0]+0.1*h
                        auxbb[2]=det["bbox"][2]-0.1*h
                    elif det["id"]==5:#right facing
                        auxbb[1]=det["bbox"][1]+0.1*w
                        auxbb[3]=det["bbox"][3]-0.1*w
                        auxbb[0]=det["bbox"][0]+0.1*h
                        auxbb[2]=det["bbox"][2]-0.1*h
                    elif det["id"]==6:#left facing
                        auxbb[1]=det["bbox"][1]#+0.1*w
                        auxbb[3]=det["bbox"][3]-0.2*w
                        auxbb[0]=det["bbox"][0]+0.1*h
                        auxbb[2]=det["bbox"][2]-0.1*h
                    elif det["id"]==7:#right facing
                        auxbb[1]=det["bbox"][1]+0.2*w
                        auxbb[3]=det["bbox"][3]#-0.1*w
                        auxbb[0]=det["bbox"][0]+0.2*h
                        auxbb[2]=det["bbox"][2]-0.1*h
                else:
                    auxbb[1]=det["bbox"][1]+0.1*w
                    auxbb[3]=det["bbox"][3]-0.1*w
                    auxbb[0]=det["bbox"][0]+0.1*h
                    auxbb[2]=det["bbox"][2]-0.1*h
                res[idd]["bbox"]=auxbb#*(1/cfg.resize)
        if show:
            showmax=30
            scrthr=0
            im=myimread(arg[ii]["file"],resize=cfg.resize)
            if tsImages[ii]["bbox"]!=[]:
                auxbb=[list(numpy.array(x)*cfg.resize) for x in tsImages[ii]["bbox"]]               
                #detectCRF.visualize2(res[:showmax],cfg.N,im,bb=auxbb,line=True,color=["w","w","w"])
                detectCRF.visualize2(res[:showmax],cfg.N,im,line=True,color=["w"]*showmax,lw=4,thr=scrthr)
            else:
                detectCRF.visualize2(res[:showmax],cfg.N,im)
            print [x["scr"] for x in res[:showmax]]
            #raw_input()
            if project:
                import extra
                gtann={"bbox":tsImages[ii]["bbox"],"facial":tsImages[ii]["facial"]}
                fpts,fid=extra.project(res,gtann,cfg.N)
                print fpts,fid
                for idl,l in enumerate(fid):
                    pylab.figure(100+l%cfg.numcl)
                    if fid<cfg.numcl:
                        fpts[idl][:,1]=models[idl]["ww"][0].shape[1]*(cfg.N/float(cfg.N+2*cfg.E))-fpts[idl][:,1]
                    pylab.plot(fpts[idl][:,1]*15,fpts[idl][:,0]*15,"or-",markersize=7)  
                    pylab.draw()
                    pylab.show()
                    raw_input()
        facial=False
        fulldist=[]
        if facial:    #evaluate facial features position
            for iddd,dd in enumerate(res):
                aflip=[1,1,1,1,1,1,1,1]
                if cfg.numcl==1:
                    anchor=numpy.array([5,5, 5,11, 9,7, 11,6, 11,8, 11,11])/8.0*models[0]["ww"][0].shape[0]/float(cfg.E*2+cfg.N)#models[res[0]["id"]]["facial"]
                if cfg.numcl==2:
                    if 0:#%cfg.db=="MultiPIE2":
                        if dd["id"]%2==0:#model 0
                             anchor=numpy.array([3,3, 3,9, 7,4, 10,3, 10,5, 10,7])/8.0*models[0]["ww"][0].shape[0]/float(cfg.E*2+cfg.N)#models[res[0]["id"]]["facial"]
                        else:#model 1
                            #anchor=numpy.array([4,6 ,4,10 ,6,10,10,6,10,8,10,10])#models[res[0]["id"]]["facial"]
                            anchor=numpy.array([3,3, 3,6 ,7,1, 11,2, 11,3, 11,5])/8.0*models[1]["ww"][0].shape[0]/float(cfg.E*2+cfg.N)#models[res[0]["id"]]["facial"]
                    else:
                        if dd["id"]%2==0:#model 0
                             anchor=numpy.array([5,5, 5,11, 8.5,7, 11,6, 11,8, 11,11])/8.0*models[0]["ww"][0].shape[0]/float(cfg.E*2+cfg.N)#models[res[0]["id"]]["facial"]
                        else:#model 1
                            #anchor=numpy.array([4,6 ,4,10 ,6,10,10,6,10,8,10,10])#models[res[0]["id"]]["facial"]
                            anchor=numpy.array([5,6, 5,8 ,8,4, 12,6, 12,7, 12,9])/8.0*models[1]["ww"][0].shape[0]/float(cfg.E*2+cfg.N)#models[res[0]["id"]]["facial"]
                elif cfg.numcl==3:#cvpr 8x8
                    if dd["id"]%cfg.numcl==0:#model 0
                        anchor=numpy.array([5.4,5, 5,10.5, 9,7.5, 11.5,6, 11.5,8, 11,11])
                        #anchor=numpy.array([6.5,7.5, 7.0,16, 12,10.5, 15.5,8, 15.5,11, 15.5,14])#/11.0*models[0]["ww"][0].shape[0]/float(cfg.E*2+cfg.N)#models[res[0]["id"]]["facial"]
                    elif dd["id"]%cfg.numcl==1:#model 1
                        anchor=numpy.array([5,5, 5,10, 8.5,5.5, 11,5.5, 11,7, 11,9.5])
                        #anchor=numpy.array([6.5,6, 6,13 ,11,10, 13.5,8, 13.5,10, 13.5,12])#/10.0*models[1]["ww"][0].shape[0]/float(cfg.E*2+cfg.N)#models[res[0]["id"]]["facial"]
                    elif dd["id"]%cfg.numcl==2:#model 2
                        anchor=numpy.array([5.5,7.5, 4.5,10.5 ,8.5,11, 11.5,8.0, 11.5,9.5, 11,10])

                elif cfg.numcl==4:
                    #aflip=[1,1,1,0] #for cvpr 8x8
                    aflip=[1,1,1,1] # for previous
                    if dd["id"]%cfg.numcl==0:#model 0
                        anchor=numpy.array([6.5,7.5, 8.0,16, 12.5,11, 15,8, 15.5,11, 16,14])
                        #anchor=numpy.array([6.5,7.5, 7.0,16, 12,10.5, 15.5,8, 15.5,11, 15.5,14])#/11.0*models[0]["ww"][0].shape[0]/float(cfg.E*2+cfg.N)#models[res[0]["id"]]["facial"]
                    elif dd["id"]%cfg.numcl==1:#model 1
                        anchor=numpy.array([7.5,6, 6,13 ,11.5,9, 14.5,7.5, 14,10.5, 13.5,13.5])
                        #anchor=numpy.array([6.5,6, 6,13 ,11,10, 13.5,8, 13.5,10, 13.5,12])#/10.0*models[1]["ww"][0].shape[0]/float(cfg.E*2+cfg.N)#models[res[0]["id"]]["facial"]
                    elif dd["id"]%cfg.numcl==2:#model 2
                        anchor=numpy.array([6.5,6.0, 6.5,13 ,11,7.5, 14,7.5, 14.5,9, 14,12.5])
                        #anchor=numpy.array([6.5,6.5, 6.5,12.5 ,11,7.5, 13.5,8, 14,10, 13.5,12.5])#/10.0*models[1]["ww"][0].shape[0]/float(cfg.E*2+cfg.N)#models[res[0]["id"]]["facial"]
                    elif dd["id"]%cfg.numcl==3:#model 3
                        #anchor=numpy.array([6,8, numpy.nan,numpy.nan ,9.5,13, 13,8, 13,10.5, numpy.nan,numpy.nan])#with occlusions
                        anchor=numpy.array([6,8, 6,12 ,9.5,13, 13,8, 13,10.5, 13,11.0])
                        #anchor=numpy.array([6.5,8.5, 5.5,10.5 ,10,13, 13,7, 13,10.5, 13,11.5])#/9.0*models[1]["ww"][0].shape[0]/float(cfg.E*2+cfg.N)#models[res[0]["id"]]["facial"]
                if dd["id"]/cfg.numcl==aflip[dd["id"]%cfg.numcl]:#flipped model
                    #inv=[14,15, 12,13, 6,7, 4,5, 8,9, 10,11, 2,3, 0,1, 18,19, 16,17]
                    inv=[2,3, 0,1, 4,5, 10,11, 8,9, 6,7] #2,3, 0,1, 18,19, 16,17]
                    #inv=[ 0,1, 2,3, 4,5, 6,7, 8,9, 10,11] #2,3, 0,1, 18,19, 16,17]
                    anchor=anchor[inv]
                    anchor[1::2]=-anchor[1::2]+int(models[dd["id"]]["ww"][0].shape[1]*float(cfg.N)/(cfg.N+2*cfg.E))
                fpoints=tsImages[ii]["facial"]
                tmpdist=[]
                for idfp,fp in enumerate(fpoints):
                #print fp;raw_input()
                    #for bb in tsImages[ii]["bbox"]
                    py1,px1,py2,px2=numpy.array(tsImages[ii]["bbox"][idfp][0:4])*cfg.resize
                    h=py2-py1
                    w=px2-px1
                    fp=fp.flatten()*cfg.resize
                    gtfp=fp.copy()
                    gtfp[0:-1:2]=fp[1::2]
                    gtfp[1::2]=fp[0:-1:2]
                    #gtfp=gtfp.flatten()
                    #pylab.plot(gtfp[1::2],gtfp[0:-1:2],"or-", markersize=8)
                    if 1:
                        efp=numpy.array(locatePoints([dd],cfg.N,anchor)[0])#,6,7,6,10,
                        efpi=numpy.array(locatePointsInter([dd],cfg.N,anchor)[0])#,6,7,6,10,
                        #print "Estimated",efp
                        #print "Ground Truth",py+fp[1],px+fp[0]
                        if iddd<showmax and dd["scr"]>scrthr:
                            #pylab.plot(efp[1::2],efp[0:-1:2],"ob-",markersize=9)
                            #pylab.plot(efpi[1::2],efpi[0:-1:2],"sg-",markersize=10)
                            pylab.plot(efpi[1::2],efpi[0:-1:2],"or",markersize=10)
                        #intocu=numpy.sqrt(((gtfp[13]+gtfp[15])/2.0-(gtfp[1]+gtfp[3])/2.0)**2+((gtfp[12]+gtfp[14])/2.0-(gtfp[0]+gtfp[2])/2.0)**2)
                        intocu=(h+w)/2.0
                        dist=numpy.sqrt(numpy.sum(numpy.reshape(gtfp-efp,(-1,2))**2,1))
                        disti=numpy.sqrt(numpy.sum(numpy.reshape(gtfp-efpi,(-1,2))**2,1))
                        #print "Pupil Left",(gtfp[1]+gtfp[3])/2,"Pupil Right",(gtfp[13]+gtfp[15])/2
                        tmpdist.append(disti)
                        res[iddd]["facial"]=efpi
                        #print "Threshold Pixels",intocu*0.05
                        #print "Dist Near",dist
                        #print "Dist Inter",disti
                        #print "Nearest",numpy.sum(dist)
                        #print "Inter",numpy.sum(disti)
                        if 0:
                            gp=numpy.sum(dist<0.05*intocu)
                            gpi=numpy.sum(disti<0.05*intocu)
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
                bestdist=numpy.argmin(numpy.array(tmpdist).sum(1))
                fulldist.append(tmpdist[bestdist])
            print "Smaller Distance",fulldist[-1]
            #raw_input()
            if type(save)==bool and save:
                pylab.savefig("save/%05d.png"%ii)
        ltdet+=res

    #raw_input()
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
    #dsfs

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
        util.save(testname+".det",{"det":ltdet})#takes a lot of space use only first 500
        util.savemat(testname+".mat",{"tp":tp,"fp":fp,"scr":scr,"tot":tot,"rc":rc,"pr":pr,"ap":ap,"dist":fulldist})
        pylab.savefig(testname+".png")
    return ap


#use a different number of hypotheses
def test(x):
    return detectCRF.test(x,show=False,inclusion=False,onlybest=False) #in bicycles is 

def testINC(x):
    return detectCRF.test(x,show=False,inclusion=True,onlybest=True) #in bicycles is better and faster with 1 hypotheses

def testINC03(x):
    return detectCRF.test(x,show=False,inclusion=True,onlybest=True,ovr=0.3) #in bicycles is better and faster with 1 hypotheses

def testINC04(x):
    return detectCRF.test(x,show=False,inclusion=True,onlybest=True,ovr=0.4) #in bicycles is better and faster 

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
    #cfg.db="mulitPIE2"
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
        tsImages=getRecord(AFW(basepath=cfg.dbpath),cfg.maxpos,facial=True)
        tsImagesFull=tsImages
    elif cfg.db=="images":
        tsImages=getRecord(DirImages(imagepath="/users/visics/mpederso/code/face-release1.0-basic/images/",ext="jpg"))
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
    #testname="data/full/face2_pose_full9"
    #testname="data/aflw/pose/face3_pose57"
    #testname="data/aflw/pose/face2_FULL7"#best results
    #testname="data/aflw/pose/face2_FULLinv4"
    #testname="data/aflw/pose/face2_FULLHIGH1"
    #testname="data/aflw/pose4/face4_hpose4_bis9"
    #testname="data/aflw/pose4/face4_FULLinvhigh2"
    #testname="data/aflw/pose4/face4_moreregdef1"
    #testname="data/aflw/pose4/face4_moreregdef11"
    #testname="data/MultiPIE3/face6_testr10_m60"
    testname="data/MultiPIE2/face2_2_2_63"
    #testname="data/cvpr/face3_AFWlow3noposefull2"
    #testname="data/cvpr/face3_AFW90001"
    #testname="data//MultiPIE/face4_fullPIE42"
    #testname="data/cvpr/face2_AFWlow23"
    #testname="data/cvpr/face4_AFWlow43"
    #testname="data/cvpr/face8_AFWlow82"
    #testname="data/MultiPIE2/face3_FULL90012"
    #testname="data/MultiPIE2/face4_FULLPIE43"
    #testname="data/MultiPIE2/face3_realFULL900_final"
    #testname="data/aflw/pose/face2_FULL9"#best results
    #testname="data/MultiPIE2/face2_highrs210"
    #testname="data/MultiPIE2/face6_FULLPIE87"
    #testname="/users/visics/mpederso/code/git/bigger/CRFdet/data/test/face1_lfw_highres_final"
    cfg.trunc=1
    models=util.load("%s.model"%(testname))
    cfg.numhypTEST=50
    #del models[0]
    cfg.numcl=len(models)/2#2#4
    cfg.E=1
    #cfg.N=2
    #cfg.sbin=4
    cfg.resize=1.0#0.5
    cfg.N=models[0]["N"]
    cfg.hallucinate=0
    if 0:
        import model
        import drawHOG
        import extra
        for idm,m in enumerate(models):
            m["ww"][0]=model.mreduce(m["ww"][0],cfg.N,cfg.E,nthr=0.08)#0.08)   
            models[idm]["cost"]=model.dreduce(m["cost"],cfg.N,cfg.E,nthr=0.02)
            im=drawHOG.drawHOG(model.convert2(m["ww"][0],cfg.N,cfg.E))
            pylab.figure(idm)
            pylab.imshow(im)
            pylab.figure(30+idm)
            extra.showDefNodes2(m["cost"])
            pylab.show()
            pylab.draw()
        raw_input()
    #    m["cost"]=m["cost"]*100
    #del models[0]["thr"]
    #models[1]["thr"]=10
    #del models[2]["thr"]
    #del models[3]["thr"]
    #del models[4]["thr"]
    #models[5]["thr"]=10
    #del models[6]["thr"]
    #del models[7]["thr"]   
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
    #runtest(models,tsImagesFull[196:197],cfg,parallel=True,numcore=16,show=True,detfun=testINC03)
    #runtest(models,tsImagesFull,cfg,parallel=True,numcore=2,show=True,detfun=testINC03,save=True)#,save="./face4_moredef1_inc04_occl")
    runtest(models,tsImagesFull,cfg,parallel=True,numcore=1,show=True,detfun=testINC,save="./face2_2_2_63")#removed 1st cluster

