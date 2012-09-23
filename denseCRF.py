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
import crf3

########################## load configuration parametes

print "Loading defautl configuration config.py"
from config import * #default configuration      

if len(sys.argv)>2: #specific configuration
    print "Loading configuration from %s"%sys.argv[2]
    import_name=sys.argv[2]
    exec "from config_%s import *"%import_name  
    #save a local configuration copy 
    import shutil
    shutil.copyfile("config_"+import_name+".py",cfg.testpath+cfg.cls+"%d"%cfg.numcl+"_"+cfg.testspec+".cfg.py")

cfg.cls=sys.argv[1]
testname=cfg.testpath+cfg.cls+("%d"%cfg.numcl)+"_"+cfg.testspec
#cfg.useRL=False#for the moment
cfg.show=False
cfg.auxdir=""
cfg.numhyp=5
cfg.numneg= 10
bias=cfg.bias
#cfg.bias=bias
cfg.posovr= 0.75
cfg.perc=0.15
#just for a fast test
#cfg.maxpos = 50
#cfg.maxneg = 20
#cfg.maxexamples = 10000
#cfg.maxtest = 20#100
parallel=True
cfg.show=False
cfg.neginpos=False
localshow=True
numcore=cfg.multipr
notreg=0
cfg.numcl=3
#cfg.valreg=0.01#set in configuration
#cfg.useRL=True

mypool = Pool(numcore) #keep the child processes as small as possible 

########################load training and test samples
if cfg.db=="VOC":
    if cfg.year=="2007":
        trPosImages=getRecord(VOC07Data(select="pos",cl="%s_trainval.txt"%cfg.cls,
                        basepath=cfg.dbpath,#"/home/databases/",
                        usetr=True,usedf=False),cfg.maxpos)
        trPosImagesNoTrunc=getRecord(VOC07Data(select="pos",cl="%s_trainval.txt"%cfg.cls,
                        basepath=cfg.dbpath,#"/home/databases/",
                        usetr=False,usedf=False),cfg.maxpos)
        trNegImages=getRecord(VOC07Data(select="neg",cl="%s_trainval.txt"%cfg.cls,
                        basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                        usetr=True,usedf=False),cfg.maxneg)
        trNegImagesFull=getRecord(VOC07Data(select="neg",cl="%s_trainval.txt"%cfg.cls,
                        basepath=cfg.dbpath,usetr=True,usedf=False),5000)
        #test
        tsPosImages=getRecord(VOC07Data(select="pos",cl="%s_test.txt"%cfg.cls,
                        basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                        usetr=True,usedf=False),cfg.maxtest)
        tsNegImages=getRecord(VOC07Data(select="neg",cl="%s_test.txt"%cfg.cls,
                        basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                        usetr=True,usedf=False),cfg.maxneg)
        tsImages=numpy.concatenate((tsPosImages,tsNegImages),0)
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

########################compute aspect ratio and dector size 
name,bb,r,a=extractInfo(trPosImages)
trpos={"name":name,"bb":bb,"ratio":r,"area":a}
import scipy.cluster.vq as vq
numcl=cfg.numcl
perc=cfg.perc#10
minres=10
minfy=3
minfx=3
#number of maximum number of HOG blocks (HOG cells /4) to use
#maxArea=45#*(4-cfg.lev[0])#too high resolution very slow
#maxArea=35#*(4-cfg.lev[0]) #the right trade-off
maxArea=25#*(4-cfg.lev[0]) #used in the test
#maxArea=15#*(4-cfg.lev[0])
usekmeans=False

sr=numpy.sort(r)
spl=[]
lfy=[];lfx=[]
cl=numpy.zeros(r.shape)
for l in range(numcl):
    spl.append(sr[round(l*len(r)/float(numcl))])
spl.append(sr[-1])
for l in range(numcl):
    cl[numpy.bitwise_and(r>=spl[l],r<=spl[l+1])]=l
for l in range(numcl):
    print "Cluster same number",l,":"
    print "Samples:",len(a[cl==l])
    #meanA=numpy.mean(a[cl==l])/16.0/(0.5*4**(cfg.lev[l]-1))#4.0
    #meanA=numpy.mean(a[cl==l])/4.0/(4**(cfg.lev[l]-1))#4.0
    meanA=numpy.mean(a[cl==l])/64.0/4.0#(4**(cfg.lev[l]-1))#4.0
    print "Mean Area:",meanA
    sa=numpy.sort(a[cl==l])
    #minA=numpy.mean(sa[len(sa)/perc])/16.0/(0.5*4**(cfg.lev[l]-1))#4.0
    #minA=numpy.mean(sa[int(len(sa)*perc)])/4.0/(4**(cfg.lev[l]-1))#4.0
    minA=numpy.mean(sa[int(len(sa)*perc)])/64.0/4.0#(4**(cfg.lev[l]-1))#4.0
    print "Min Area:",minA
    aspt=numpy.mean(r[cl==l])
    print "Aspect:",aspt
    if minA>maxArea:
        minA=maxArea
    #minA=10#for bottle
    if aspt>1:
        fx=(max(minfx,numpy.sqrt(minA/aspt)))
        fy=(fx*aspt)
    else:
        fy=(max(minfy,numpy.sqrt(minA*(aspt))))
        fx=(fy/(aspt))        
    print "Fy:%.2f"%fy,"~",round(fy),"Fx:%.2f"%fx,"~",round(fx)
    lfy.append(round(fy))
    lfx.append(round(fx))
    print

#raw_input()

cfg.fy=lfy#[7,10]#lfy
cfg.fx=lfx#[11,7]#lfx
# the real detector size would be (cfg.fy,cfg.fx)*2 hog cells


############################ initialize positive using cropped bounidng boxes
check = False
import pylab as pl
dratios=numpy.array(cfg.fy)/numpy.array(cfg.fx)
hogp=[[] for x in range(cfg.numcl)]
hogpcl=[]
annp=[[] for x in range(cfg.numcl)]

#from scipy.ndimage import zoom
from extra import myzoom as zoom
for im in trPosImagesNoTrunc: # for each image

    aim=util.myimread(im["name"])  
    for bb in im["bbox"]: # for each bbox (y1,x1,y2,x2)
        imy=bb[2]-bb[0]
        imx=bb[3]-bb[1]
        cropratio= imy/float(imx)
        #select the right model based on aspect ratio
        idm=numpy.argmin(abs(dratios-cropratio))
        crop=aim[max(0,bb[0]-imy/cfg.fy[idm]/2):min(bb[2]+imy/cfg.fy[idm]/2,aim.shape[0]),max(0,bb[1]-imx/cfg.fx[idm]/2):min(bb[3]+imx/cfg.fx[idm]/2,aim.shape[1])]
        #crop=extra.getfeat(aim,abb[0]-imy/(cfg.fy[idm]*2),bb[2]+imy/(cfg.fy[idm]*2),bb[1]-imx/(cfg.fx[idm]*2),bb[3]+imx/(cfg.fx[idm]*2))
        imy=crop.shape[0]
        imx=crop.shape[1]
        zcim=zoom(crop,(((cfg.fy[idm]*2+2)*8/float(imy)),((cfg.fx[idm]*2+2)*8/float(imx)),1),order=1)
        hogp[idm].append(numpy.ascontiguousarray(pyrHOG2.hog(zcim)))
        #hogpcl.append(idm)
        annp[idm].append({"file":im["name"],"bbox":bb})
        if check:
            print "Aspect:",idm,"Det Size",cfg.fy[idm]*2,cfg.fx[idm]*2,"Shape:",zcim.shape
            pl.figure(1,figsize=(20,5))
            pl.clf()
            pl.subplot(1,3,1)
            pl.imshow(aim,interpolation="nearest")            
            pl.subplot(1,3,2)
            pl.imshow(zcim,interpolation="nearest")
            pl.subplot(1,3,3)
            import drawHOG
            imh=drawHOG.drawHOG(hogp[-1])
            pl.imshow(imh,interpolation="nearest")
            pl.draw()
            pl.show()
            raw_input()
for l in range(cfg.numcl):
    print "Aspect",l,":",len(hogp[l]),"samples"
    hogpcl=hogpcl+[l]*len(hogp[l])    

# get some random negative images
hogn=[[] for x in range(cfg.numcl)]
hogncl=[]
check=False
numpy.random.seed(3) # to reproduce results
#from scipy.ndimage import zoom
for im in trNegImages: # for each image
    aim=util.myimread(im["name"])
    for idm in range(cfg.numcl):
        szy=(cfg.fy[idm]*2+2)
        szx=(cfg.fx[idm]*2+2)
        rndy=numpy.random.randint(0,aim.shape[0]-szy*8-1)
        rndx=numpy.random.randint(0,aim.shape[1]-szx*8-1)
        #zcim=zoom(crop,(((cfg.fy[idm]*2+2)*8/float(imy)),((cfg.fx[idm]*2+2)*8/float(imx)),1),order=1)
        zcim=aim[rndy:rndy+szy*8,rndx:rndx+szx*8]
        hogn[idm].append(numpy.ascontiguousarray(pyrHOG2.hog(zcim)).flatten())
        #hogncl.append(idm)
        if check:
            print "Aspcet",idm,"HOG",hogn[-1].shape
for l in range(cfg.numcl):  
    print "Aspect",l,":",len(hogn[l]),"samples"
    hogncl=hogncl+[l]*len(hogn[l])    

#################### cluster left right
trpos=[]
#trposcl=[]
if cfg.useRL:
    for l in range(numcl):
        mytrpos=[]            
        #for c in range(len(trpos)):
        #    if hogpcl[c]==l:
        for idel,el in enumerate(hogp[l]):
            if 0:#annp[l][idel]["bbox"][6]=="Left":
                faux=pyrHOG2.hogflip(el)
                #mytrpos.append(pyrHOG2.hogflip(el))    
            else:
                faux=el.copy()
                #mytrpos.append(el)
        #laux=mytrpos[:]
        #for idel,el in enumerate(laux):
            mytrpos.append((faux))            
            mytrpos.append(pyrHOG2.hogflip(faux))            
            #print idel,el.shape
            #raw_input()
        mytrpos=numpy.array(mytrpos)
        cl1=range(0,len(mytrpos),2)
        cl2=range(1,len(mytrpos),2)
        #rrnum=len(mytrpos)
        rrnum=min(len(mytrpos),1000)#to avoid too long clustering
        for rr in range(rrnum):
            print "Clustering iteration ",rr
            oldvar=numpy.sum(numpy.var(mytrpos[cl1],0))+numpy.sum(numpy.var(mytrpos[cl2],0))
            #print "Variance",oldvar
            #print "Var1",numpy.sum(numpy.var(mytrpos[cl1],0))
            #print "Var2",numpy.sum(numpy.var(mytrpos[cl2],0))
            #c1=numpy.mean(mytrpos[cl1])
            #c2=numpy.mean(mytrpos[cl1])
            rel=numpy.random.randint(len(cl1))
            tmp=cl1[rel]
            cl1[rel]=cl2[rel]
            cl2[rel]=tmp
            newvar=numpy.sum(numpy.var(mytrpos[cl1],0))+numpy.sum(numpy.var(mytrpos[cl2],0))
            if newvar>oldvar:#go back
                tmp=cl1[rel]
                cl1[rel]=cl2[rel]
                cl2[rel]=tmp
            else:
                print "Variance",newvar
        print "Elements Cluster ",l,": ",len(cl1)
        for cc in cl1:
            trpos.append(mytrpos[cc].flatten())
        #trpos+=(mytrpos[cl1]).tolist()
        #trposcl+=([l]*len(cl1))
    #flatten
    #for idl,l in enumerate(trpos):
    #    trpos[idl]=trpos[idl].flatten()
else:   
    for l in range(cfg.numcl):
        trpos+=hogp[l]
 
#################### train a first detector

#empty rigid model
models=[]
for c in range(cfg.numcl):      
    models.append(model.initmodel(cfg.fy[c]*2,cfg.fx[c]*2,1,cfg.useRL,deform=False))

#array with dimensions of w
cumsize=numpy.zeros(numcl+1,dtype=numpy.int)
for idl in range(numcl):
    cumsize[idl+1]=cumsize[idl]+(cfg.fy[idl]*2*cfg.fx[idl]*2)*31+1

try:
    fsf
    models=util.load("%s%d.model"%(testname,0))
    print "Loading Pretrained Initial detector"
except:
    # train detector
    import pegasos
    #trpos=[]
    trneg=[]
    for l in range(cfg.numcl):
        #trpos+=hogp[l]
        trneg+=hogn[l]

    w,r,prloss=pegasos.trainComp(trpos,trneg,"",hogpcl,hogncl,pc=cfg.svmc,k=1,numthr=1,eps=0.01,bias=bias)#,notreg=notreg)

    waux=[]
    rr=[]
    w1=numpy.array([])
    #from w to model m1
    for idm,m in enumerate(models):
        models[idm]=model.w2model(w[cumsize[idm]:cumsize[idm+1]-1],-w[cumsize[idm+1]-1]*bias,len(m["ww"]),31,m["ww"][0].shape[0],m["ww"][0].shape[1])
        #models[idm]["ra"]=w[cumsize[idm+1]-1]
        #from model to w #changing the clip...
        waux.append(model.model2w(models[idm],False,False,False))
        rr.append(models[idm]["rho"])
        w1=numpy.concatenate((w1,waux[-1],-numpy.array([models[idm]["rho"]])/bias))
    w2=w
    w=w1

    util.save("%s%d.model"%(testname,0),models)

#show model 
#mm=w[:1860].reshape((cfg.fy[0]*2,cfg.fx[0]*2,31))
it = 0
for idm,m in enumerate(models):   
    import drawHOG
    imm=drawHOG.drawHOG(m["ww"][0])
    pl.figure(100+idm,figsize=(3,3))
    pl.imshow(imm)
    pylab.savefig("%s_hog%d_cl%d.png"%(testname,it,idm))

pl.draw()
pl.show()    
#raw_input()

######################### add CRF and rebuild w
for idm,m in enumerate(models):   
    models[idm]["cost"]=0.01*numpy.ones((8,cfg.fy[idm],cfg.fx[idm]))

waux=[]
rr=[]
w1=numpy.array([])
sizereg=numpy.zeros(cfg.numcl,dtype=numpy.int32)
#from w to model m1
for idm,m in enumerate(models):
    waux.append(model.model2w(models[idm],False,False,False,useCRF=True,k=cfg.k))
    rr.append(models[idm]["rho"])
    w1=numpy.concatenate((w1,waux[-1],-numpy.array([models[idm]["rho"]])/bias))
    sizereg[idm]=models[idm]["cost"].size
#w2=w #old w
w=w1

#add ids clsize and cumsize for each model
clsize=[]
cumsize=numpy.zeros(numcl+1,dtype=numpy.int)
for l in range(cfg.numcl):
    models[l]["id"]=l
    clsize.append(len(waux[l])+1)
    cumsize[l+1]=cumsize[l]+len(waux[l])+1
clsize=numpy.array(clsize)

if cfg.useRL:
    #add flipped models
    for idm in range(cfg.numcl):
        models.append(extra.flip(models[idm]))
        models[-1]["id"]=idm+cfg.numcl

lndet=[] #save negative detections
lnfeat=[] #
lnedge=[] #
lndetnew=[]

lpdet=[] #save positive detections
lpfeat=[] #
lpedge=[] #

total=[]
negratio=[]
negratio2=[]

#from scipy.ndimage import zoom
import detectCRF
from multiprocessing import Pool
import itertools

####################### repeat scan positives
for it in range(cfg.posit):

    #mypool = Pool(numcore)
    #counters
    padd=0
    pbetter=0
    pworst=0
    pold=0

    arg=[]
    for idl,l in enumerate(trPosImages):
        bb=l["bbox"]
        for idb,b in enumerate(bb):
            if cfg.useRL:
                arg.append({"idim":idl,"file":l["name"],"idbb":idb,"bbox":b,"models":models,"cfg":cfg,"flip":False})    
                arg.append({"idim":idl,"file":l["name"],"idbb":idb,"bbox":b,"models":models,"cfg":cfg,"flip":True})    
            else:
                arg.append({"idim":idl,"file":l["name"],"idbb":idb,"bbox":b,"models":models,"cfg":cfg,"flip":False})    

    #lpdet=[];lpfeat=[];lpedge=[]
    if not(parallel):
        itr=itertools.imap(detectCRF.refinePos,arg)        
    else:
        itr=mypool.imap(detectCRF.refinePos,arg)

    for ii,res in enumerate(itr):
        found=False
        if res[0]!=[]:
            #compare new score with old
            newdet=res[0]
            newfeat=res[1]
            newedge=res[2]
            for idl,l in enumerate(lpdet):
                #print "Newdet",newdet["idim"],"Olddet",l["idim"]
                if (newdet["idim"]==l["idim"]): #same image
                    if (newdet["idbb"]==l["idbb"]): #same bbox
                        if (newdet["scr"]>l["scr"]):#compare score
                            print "New detection has better score"
                            lpdet[idl]=newdet
                            lpfeat[idl]=newfeat
                            lpedge[idl]=newedge
                            found=True
                            pbetter+=1
                        else:
                            print "New detection has worst score"
                            found=True
                            pworst+=1
            if not(found):
                print "Added a new sample"
                lpdet.append(res[0])
                lpfeat.append(res[1])
                lpedge.append(res[2])
                padd+=1
        else: #not found any detection with enough overlap
            for idl,l in enumerate(lpdet):
                iname=arg[ii]["file"].split("/")[-1]
                if cfg.useRL:
                    if arg[ii]["flip"]:
                        iname=iname+".flip"
                if (iname==l["idim"]): #same image
                    if (arg[ii]["idbb"]==l["idbb"]): #same bbox
                        print "Example not found, but keep old detection"                        
                        pold+=1
                        found=True
        if localshow:
            im=util.myimread(arg[ii]["file"],arg[ii]["flip"])
            rescale,y1,x1,y2,x2=res[3]
            #bb=arg[ii]["bbox"]#res[0]["bbox"]
            #dy=bb[2]-bb[0]
            #dx=bb[3]-bb[1]
            if res[0]!=[]:
                if found:
                    text="Already detected example"
                else:
                    text="Added a new example"
            else:
                if found:
                    text="Keep old detection"
                else:
                    text="No detection"
            im=extra.myzoom(im[y1:y2,x1:x2],(rescale,rescale,1),1)
            if res[0]!=[]:
                detectCRF.visualize2([res[0]],im,text)
            else:
                detectCRF.visualize2([],im,text)
            #if it>0:
                #raw_input()
    print "Added examples",padd
    print "Improved examples",pbetter
    print "Old examples score",pworst
    print "Old examples bbox",pold
    total.append(padd+pbetter+pworst+pold)
    print "Total",total,"/",len(arg)
    #raw_input()
              
    #build training data for positives
    trpos=[]
    trposcl=[]
    for idl,l in enumerate(lpdet):#enumerate(lndet):
        efeat=lpfeat[idl]#.flatten()
        eedge=lpedge[idl]#.flatten()
        if lpdet[idl]["id"]>=cfg.numcl:#flipped version
            efeat=pyrHOG2.hogflip(efeat)
            eedge=pyrHOG2.crfflip(eedge)
        trpos.append(numpy.concatenate((efeat.flatten(),eedge.flatten())))
        trposcl.append(l["id"]%cfg.numcl)
 
    ########### repeat scan negatives
    for nit in range(cfg.negit):

        ########### from detections build training data
        
        ltosort=[-x["scr"] for x in lndet]
        lord=numpy.argsort(ltosort)
        trneg=[]
        trnegcl=[]

        #filter and build negative vectors
        auxdet=[]
        auxfeat=[]
        auxedge=[]
        for idl in lord[:cfg.maxexamples/2]:#to maintain space for new samples
            auxdet.append(lndet[idl])
            auxfeat.append(lnfeat[idl])
            auxedge.append(lnedge[idl])
            efeat=lnfeat[idl]#.flatten()
            eedge=lnedge[idl]#.flatten()
            if lndet[idl]["id"]>=cfg.numcl:#flipped version
                efeat=pyrHOG2.hogflip(efeat)
                eedge=pyrHOG2.crfflip(eedge)
            trneg.append(numpy.concatenate((efeat.flatten(),eedge.flatten())))
            trnegcl.append(lndet[idl]["id"]%cfg.numcl)
            
        lndet=auxdet
        lnfeat=auxfeat
        lnedge=auxedge

        print "Total Negative Samples:",len(ltosort)
        print "New Extracted Negatives",len(lndetnew)
        print "Negative Support Vectors:",numpy.sum(-numpy.array(ltosort)>-1)
        print "Negative Cache Vectors:",len(lndet)
        print "Kept negative Support Vectors:",cfg.maxexamples
        #raw_input()

        #if no negative sample add empty negatives
        for l in range(cfg.numcl):
            if numpy.sum(numpy.array(trnegcl)==l)==0:
                trneg.append(numpy.concatenate((numpy.zeros(models[l]["ww"][0].shape).flatten(),numpy.zeros(models[l]["cost"].shape).flatten())))
                trnegcl.append(l)

        ############ check convergency
        if nit>0: # and not(limit):
            posl,negl,reg,nobj,hpos,hneg=pegasos.objective(trpos,trneg,trposcl,trnegcl,clsize,w,cfg.svmc,cfg.bias,sizereg=sizereg,valreg=cfg.valreg)#,notreg)
            print "NIT:",nit,"OLDLOSS",old_nobj,"NEWLOSS:",nobj
            negratio.append(nobj/(old_nobj+0.000001))
            negratio2.append((posl+negl)/(old_posl+old_negl+0.000001))
            print "RATIO: newobj/oldobj:",negratio,negratio2
            output="Negative not converging yet!"
            #if (negratio[-1]<1.05):
            if (negratio[-1]<cfg.convNeg):
                print "Very small invrement of loss: negative convergence at iteration %d!"%nit
                break

        ############train a new detector with new positive and all negatives
        #print elements per model
        for l in range(cfg.numcl):
            print "Model",l
            print "Positive Examples:",numpy.sum(numpy.array(trposcl)==l)
            print "Negative Examples:",numpy.sum(numpy.array(trnegcl)==l)
    
        import pegasos   
        w,r,prloss=pegasos.trainComp(trpos,trneg,"",trposcl,trnegcl,oldw=w,pc=cfg.svmc,k=numcore*2,numthr=numcore,eps=0.01,bias=cfg.bias,sizereg=sizereg,valreg=cfg.valreg)#,notreg=notreg)

        pylab.figure(300,figsize=(4,4))
        pylab.clf()
        pylab.plot(w)
        pylab.title("dimensions of W")
        pylab.draw()
        pylab.show()
        #raw_input()

        old_posl,old_negl,old_reg,old_nobj,old_hpos,old_hneg=pegasos.objective(trpos,trneg,trposcl,trnegcl,clsize,w,cfg.svmc,cfg.bias,sizereg=sizereg,valreg=cfg.valreg)#,notreg) 
        waux=[]
        rr=[]
        w1=numpy.array([])
        #from w to model m1
        for idm,m in enumerate(models[:cfg.numcl]):
            models[idm]=model.w2model(w[cumsize[idm]:cumsize[idm+1]-1],-w[cumsize[idm+1]-1]*bias,len(m["ww"]),31,m["ww"][0].shape[0],m["ww"][0].shape[1],useCRF=True,k=cfg.k)
            models[idm]["id"]=idm
            #models[idm]["ra"]=w[cumsize[idm+1]-1]
            #from model to w #changing the clip...
            waux.append(model.model2w(models[idm],False,False,False,useCRF=True,k=cfg.k))
            #rr.append(models[idm]["rho"])
            w1=numpy.concatenate((w1,waux[-1],-numpy.array([models[idm]["rho"]])/bias))
        w2=w
        w=w1

        if cfg.useRL:
            #add flipped models
            for idm in range(cfg.numcl):
                models[cfg.numcl+idm]=(extra.flip(models[idm]))
                models[cfg.numcl+idm]["id"]=idm+cfg.numcl


        util.save("%s%d.model"%(testname,it),models)

        #visualize models
        for idm,m in enumerate(models[:cfg.numcl]):   
            import drawHOG
            imm=drawHOG.drawHOG(m["ww"][0])
            pl.figure(100+idm,figsize=(3,3))
            pl.clf()
            pl.imshow(imm)
            pl.title("b:%.3f h:%.4f d:%.4f"%(m["rho"],numpy.sum(m["ww"][0]**2),numpy.sum(m["cost"]**2)))
            pl.draw()
            pl.show()
            pylab.savefig("%s_hog%d_cl%d.png"%(testname,it,idm))
            #CRF
            pl.figure(110+idm,figsize=(5,5))
            pl.clf()
            extra.showDef(m["cost"][:4])
            pl.draw()
            pl.show()
            pylab.savefig("%s_def%dl_cl%d.png"%(testname,it,idm))
            pl.figure(120+idm,figsize=(5,5))
            pl.clf()
            extra.showDef(m["cost"][4:])
            pl.draw()
            pl.show()
            pylab.savefig("%s_def%dq_cl%d.png"%(testname,it,idm))

        ########### scan negatives
        lndetnew=[];lnfeatnew=[];lnedgenew=[]
        arg=[]
        for idl,l in enumerate(trNegImages):
            #bb=l["bbox"]
            #for idb,b in enumerate(bb):
            arg.append({"idim":idl,"file":l["name"],"idbb":0,"bbox":[],"models":models,"cfg":cfg,"flip":False})    

        if not(parallel):
            itr=itertools.imap(detectCRF.hardNeg,arg)        
        else:
            itr=mypool.imap(detectCRF.hardNeg,arg)

        for ii,res in enumerate(itr):
            if localshow:
                im=myimread(arg[ii]["file"])
                detectCRF.visualize2(res[0],im)
            lndetnew+=res[0]
            lnfeatnew+=res[1]
            lnedgenew+=res[2]
        ########### scan negatives in positives
        
        if cfg.neginpos:
            arg=[]
            for idl,l in enumerate(trPosImages):
                #bb=l["bbox"]
                #for idb,b in enumerate(bb):
                arg.append({"idim":idl,"file":l["name"],"idbb":0,"bbox":l["bbox"],"models":models,"cfg":cfg,"flip":False})    

            lndetnew=[];lnfeatnew=[];lnedgenew=[]
            if not(parallel):
                itr=itertools.imap(detectCRF.hardNegPos,arg)        
            else:
                itr=mypool.imap(detectCRF.hardNegPos,arg)

            for ii,res in enumerate(itr):
                if localshow:
                    im=myimread(arg[ii]["file"])
                    detectCRF.visualize2(res[0],im)
                lndetnew+=res[0]
                lnfeatnew+=res[1]
                lnedgenew+=res[2]

        ########## rescore old negative detections
        for idl,l in enumerate(lndet):
            idm=l["id"]
            lndet[idl]["scr"]=numpy.sum(models[idm]["ww"][0]*lnfeat[idl])+numpy.sum(models[idm]["cost"]*lnedge[idl])-models[idm]["rho"]#-rr[idm]/bias

        ########## rescore old positive detections
        for idl,l in enumerate(lpdet):
            idm=l["id"]
            lpdet[idl]["scr"]=numpy.sum(models[idm]["ww"][0]*lpfeat[idl])+numpy.sum(models[idm]["cost"]*lpedge[idl])-models[idm]["rho"]#-rr[idm]/bias

        ########### include new detections in the old pool discarding doubles
        #auxdet=[]
        #auxfeat=[]
        #aux=[]
        for newid,newdet in enumerate(lndetnew): # for each newdet
            #newdet=ldetnew[newid]
            remove=False
            for oldid,olddet in enumerate(lndet): # check with the old
                if (newdet["idim"]==olddet["idim"]): #same image
                    if (newdet["scl"]==olddet["scl"]): #same scale
                        if (newdet["id"]==olddet["id"]): #same model
                            if (numpy.all(newdet["def"]==olddet["def"])): #same deformation
                                #same features
                                assert(newdet["scr"]-olddet["scr"]<0.00001)
                                assert(numpy.all(lnfeatnew[newid]-lnfeat[oldid]<0.00001))
                                assert(numpy.all(lnedgenew[newid]==lnedge[oldid]))
                                print "Detection",newdet["idim"],newdet["scl"],newdet["id"],"is double --> removed!"
                                remove=True
            if not(remove):
                lndet.append(lndetnew[newid])
                lnfeat.append(lnfeatnew[newid])
                lnedge.append(lnedgenew[newid])
                
    #mypool.close()
    #mypool.join()
    ##############test
    import denseCRFtest
    #denseCRFtest.runtest(models,tsImages,cfg,parallel=True,numcore=numcore,save="%s%d"%(testname,it),detfun=lambda x :detectCRF.test(x,numhyp=1,show=False),show=localshow)

    denseCRFtest.runtest(models,tsImages,cfg,parallel=True,numcore=numcore,save="%s%d"%(testname,it),show=localshow,pool=mypool,detfun=denseCRFtest.test1hypINC)


# unitl positve convergercy

