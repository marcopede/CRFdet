#function to detect using the CRF approach

import util
import numpy
import pylab
import pyrHOG2
import crf3
import drawHOG
import time
import copy
#from scipy.ndimage import zoom
from extra import myzoom as zoom

def refinePos(el):
    t=time.time()
    rescale=1.0
    #[f,det]=detectCRF.detectCrop(el)
    print "----Image-%s-(%d)-----------"%(el["file"].split("/")[-1],el["idim"])
    imname=el["file"]
    bbox=el["bbox"]
    models=el["models"]
    cfg=el["cfg"]
    imageflip=el["flip"]
    dratios=[]
    fy=[]
    fx=[]
    det=[]
    for m in models:
        fy.append(m["ww"][0].shape[0]/cfg.N)
        fx.append(m["ww"][0].shape[1]/cfg.N)
        dratios.append(fy[-1]/float(fx[-1]))
    fy=numpy.array(fy)
    fx=numpy.array(fx)
    dratios=numpy.array(dratios)
    #dratios=numpy.array(cfg.fy)/numpy.array(cfg.fx)
    img=util.myimread(imname,resize=cfg.resize)
    #make a mirror out of the image --> max out 50%
    print "Old bbox:",bbox
    extbbox=numpy.array(bbox[:6])
    if bbox[0]<=1 and cfg.useclip:
        extbbox[0]=-bbox[2]
        extbbox[4]=1
    if bbox[1]<=1 and cfg.useclip:
        extbbox[1]=-bbox[3]
        extbbox[4]=1
    if bbox[2]>=img.shape[0]-1 and cfg.useclip:
        extbbox[2]=img.shape[0]+(img.shape[0]-bbox[0])
        extbbox[4]=1
    if bbox[3]>=img.shape[1]-1 and cfg.useclip:
        extbbox[3]=img.shape[1]+(img.shape[1]-bbox[1])
        extbbox[4]=1
    print "New bbox:",extbbox
    #raw_input()
    if imageflip:
        img=util.myimread(imname,True,resize=cfg.resize)
        if bbox!=None:
            bbox = util.flipBBox(img,[bbox])[0]
            extbbox = util.flipBBox(img,[extbbox])[0]    
    maxy=numpy.max([x["ww"][0].shape[0] for x in models])    
    maxx=numpy.max([x["ww"][0].shape[1] for x in models])    
    if (bbox!=[]):
        marginy=(bbox[2]-bbox[0])*0.5 #safe margin to be sure the border is not used
        marginx=(bbox[3]-bbox[1])*0.5
        y1=max(0,bbox[0]-marginy);x1=max(0,bbox[1]-marginx)
        y2=min(bbox[2]+marginy,img.shape[0]);x2=min(bbox[3]+marginx,img.shape[1])
        img=img[y1:y2,x1:x2]
        newbbox=(bbox[0]-y1,bbox[1]-x1,bbox[2]-y1,bbox[3]-x1)
        extnewbbox=(extbbox[0]-y1,extbbox[1]-x1,extbbox[2]-y1,extbbox[3]-x1)
        cropratio= marginy/float(marginx)
        dist=abs(numpy.log(dratios)-numpy.log(cropratio))
        #idm=numpy.where(dist<0.4)[0] #
        idm=numpy.where(dist<0.5)[0] #
        #print "                 Selected ratios",idm
        if cfg.useclip:
            if bbox[4]==1 or extbbox[4]==1:# use all models for truncated
                #print "TRUNCATED!!!"
                idm=range(len(models)) 
        if len(idm)>0:
            if cfg.rescale:# and len(idm)>0:
                #tiley=((bbox[2]-bbox[0]))/float(numpy.max(fy[idm]))
                #tilex=((bbox[3]-bbox[1]))/float(numpy.max(fx[idm]))
                tiley=((extbbox[2]-extbbox[0]))/float(numpy.max(fy[idm]))
                tilex=((extbbox[3]-extbbox[1]))/float(numpy.max(fx[idm]))
                #print "Tile y",tiley,(bbox[2]-bbox[0])/8,"Tile x",tilex,(bbox[3]-bbox[1])/8
                if tiley>8*cfg.N and tilex>8*cfg.N:
                    rescale=(8*cfg.N)/float(min(tiley,tilex))
                    img=zoom(img,(rescale,rescale,1),order=1)
                    newbbox=numpy.array(newbbox)*rescale
                    extnewbbox=numpy.array(extnewbbox)*rescale
                else:
                    print "Not Rescaling!!!!"
            selmodels=[models[x] for x in idm] 
            #t1=time.time()
            if cfg.usebbPOS:
                [f,det]=rundetbb(img,cfg.N,selmodels,numdet=cfg.numhypPOS, interv=cfg.intervPOS,aiter=cfg.aiterPOS,restart=cfg.restartPOS,trunc=cfg.trunc)
            else:         
                #[f,det]=rundet(img,cfg.N,selmodels,numhyp=cfg.numhypPOS,interv=cfg.intervPOS,aiter=cfg.aiterPOS,restart=cfg.restartPOS,trunc=cfg.trunc)
                [f,det]=rundetc(img,cfg.N,selmodels,numhyp=cfg.numhypPOS,interv=cfg.intervPOS,aiter=cfg.
aiterPOS,restart=cfg.restartPOS,trunc=cfg.trunc,bbox=extnewbbox)
    #else: #for negatives
    #    [f,det]=rundet(img,cfg,models)
    #print "Rundet time:",time.time()-t1
    print "Detection time:",time.time()-t
    #detectCRF.check(det[:10],f,models)
    boundingbox(det,cfg.N)
    if cfg.useclip:
        clip(det,img.shape)
    #detectCRF.visualize(det[:10],f,img,cfgpos)
    bestscr=-100
    best=-1
    for idl,l in enumerate(det):
        ovr=util.overlap(newbbox,l["bbox"])
        #print ovr,l["scr"]
        if ovr>cfg.posovr and l["scr"]>cfg.posthr:#valid detection
            if l["scr"]>bestscr:
                best=idl
                bestscr=l["scr"]
    #raw_input()
    if len(det)>0 and best!=-1:
        print "Pos det:",[x["scr"] for x in det[:5]]
        if cfg.show:
            visualize([det[best]],cfg.N,f,img)
        feat,edge=getfeature([det[best]],cfg.N,f,models,cfg.trunc)
        #add image name and bbx so that each annotation is unique
        if imageflip:
            det[best]["idim"]=el["file"].split("/")[-1]+".flip"
        else:
            det[best]["idim"]=el["file"].split("/")[-1]
        det[best]["idbb"]=el["idbb"]
        #add bias
        #det[best]["scr"]-=models[det[best]["id"]]["rho"]/float(cfg.bias)
        return det[best],feat[0],edge[0],[rescale,y1,x1,y2,x2]#last just for drawing
    return [],[],[],[rescale,y1,x1,y2,x2]



def hardNeg(el):
    t=time.time()
    #[f,det]=detectCRF.detectCrop(el)
    print "----Image-%s-(%d)-----------"%(el["file"].split("/")[-1],el["idim"])
    imname=el["file"]
    bbox=el["bbox"]
    models=el["models"]
    cfg=el["cfg"]
    #cfg.numhyp=1 #make the search for negatives 5 times faster
    #                #but you should check if generates problems
    imageflip=el["flip"]
    if imageflip:
        img=util.myimread(imname,True,resize=cfg.resize)
    else:
        img=util.myimread(imname,resize=cfg.resize)
    #imageflip=el["flip"]
    if cfg.usebbNEG:
        [f,det]=rundetbb(img,cfg.N,models,numdet=cfg.numhypNEG,interv=cfg.intervNEG,aiter=cfg.aiterNEG,restart=cfg.restartNEG,trunc=cfg.trunc)
    else:
        [f,det]=rundet(img,cfg.N,models,numhyp=cfg.numhypNEG,interv=cfg.intervNEG,aiter=cfg.aiterNEG,restart=cfg.restartNEG,trunc=cfg.trunc)
    ldet=[]
    lfeat=[]
    ledge=[]
    for idl,l in enumerate(det[:cfg.numneg]):
        #add bias
        #det[idl]["scr"]-=models[det[idl]["id"]]["rho"]/float(cfg.bias)
        if det[idl]["scr"]>-1:
            det[idl]["idim"]=el["file"].split("/")[-1]
            ldet.append(det[idl])
            feat,edge=getfeature([det[idl]],cfg.N,f,models,cfg.trunc)
            lfeat+=feat
            ledge+=edge
    if cfg.show:
        visualize(ldet,cfg.N,f,img)
    print "Detection time:",time.time()-t
    print "Found %d hard negatives"%len(ldet)
    return ldet,lfeat,ledge

def hardNegPos(el):
    t=time.time()
    #[f,det]=detectCRF.detectCrop(el)
    print "----Image-%s-(%d)-----------"%(el["file"].split("/")[-1],el["idim"])
    imname=el["file"]
    bbox=el["bbox"]
    models=el["models"]
    cfg=el["cfg"]
    #cfg.numhyp=1 #make the search for negatives 5 times faster
    #                #but you should check if generates problems
    imageflip=el["flip"]
    if imageflip:
        img=util.myimread(imname,True,resize=cfg.resize)
    else:
        img=util.myimread(imname,resize=cfg.resize)
    #imageflip=el["flip"]
    if cfg.usebbNEG:
        [f,det]=rundetbb(img,cfg.N,models,numdet=cfg.numhypNEG,interv=cfg.intervNEG,aiter=cfg.aiterNEG,restart=cfg.restartNEG,trunc=cfg.trunc)
    else:
        [f,det]=rundet(img,cfg.N,models,numhyp=cfg.numhypNEG,interv=cfg.intervNEG,aiter=cfg.aiterNEG,restart=cfg.restartNEG,trunc=cfg.trunc)
    ldet=[]
    lfeat=[]
    ledge=[]
    boundingbox(det,cfg.N)
    if cfg.useclip:
        clip(det,img.shape)
    for idl,l in enumerate(det):#(det[:cfg.numneg]):
        #add bias
        #det[idl]["scr"]-=models[det[idl]["id"]]["rho"]/float(cfg.bias)
        skip=False
        if det[idl]["scr"]>-1:
            for gt in bbox:
                ovr=util.overlap(det[idl]["bbox"],gt)
                if ovr>0.3 and (cfg.db!="inria" or ovr<0):#is not a false positive 
                    skip=True
                    break
            if not(skip):
                det[idl]["idim"]=el["file"].split("/")[-1]
                ldet.append(det[idl])
                feat,edge=getfeature([det[idl]],cfg.N,f,models,cfg.trunc)
                lfeat+=feat
                ledge+=edge
        if len(ldet)==cfg.numneg:
            break
    if cfg.show:
        visualize(ldet,cfg.N,f,img)
    print "Detection time:",time.time()-t
    print "Found %d hard negatives"%len(ldet)
    return ldet,lfeat,ledge


def test(el,docluster=True,show=False,inclusion=False,onlybest=False):
    t=time.time()
    #[f,det]=detectCRF.detectCrop(el)
    print "----Image-%s-(%d)-----------"%(el["file"].split("/")[-1],el["idim"])
    imname=el["file"]
    bbox=el["bbox"]
    models=el["models"]
    cfg=el["cfg"]
    imageflip=el["flip"]
    if imageflip:
        img=util.myimread(imname,True,resize=cfg.resize)
    else:
        img=util.myimread(imname,resize=cfg.resize)
    #imageflip=el["flip"]
    if cfg.usebbTEST:
        [f,det]=rundetbb(img,cfg.N,models,numdet=cfg.numhypTEST,interv=cfg.intervTEST,aiter=cfg.aiterTEST,restart=cfg.restartTEST,trunc=cfg.trunc)
    else:
        [f,det]=rundet(img,cfg.N,models,numhyp=cfg.numhypTEST,interv=cfg.intervTEST,aiter=cfg.aiterTEST,restart=cfg.restartTEST,trunc=cfg.trunc)
    boundingbox(det,cfg.N)
    if cfg.useclip:
        clip(det,img.shape)
    if docluster:
        #det=cluster(det,maxcl=100,inclusion=False)
        det=cluster(det,maxcl=100,inclusion=inclusion,onlybest=onlybest)
    for idl,l in enumerate(det):
        det[idl]["idim"]=el["file"].split("/")[-1]
    if show:
        visualize(det[:5],cfg.N,f,img)
    print "Detection time:",time.time()-t
    return det


def cluster(det,ovr=0.5,maxcl=20,inclusion=False,onlybest=False):
    """
    cluster detection with a minimum area k of overlapping
    """
    cllist=[]
    for ls in det:
        found=False
        for cl in cllist:
            for cle in cl:
                if not(inclusion):
                    myovr=util.overlap(ls["bbox"],cle["bbox"])
                else:   
                    myovr=util.inclusion(ls["bbox"],cle["bbox"])
                if myovr>ovr:
                    if not(onlybest):
                        cl.append(ls)
                    found=True
                    break
        if not(found):
            if len(cllist)<maxcl:
                cllist.append([ls])
            else:
                break
    return [el[0] for el in cllist]


def check(det,f,models,bias):
    """ check if score of detections and score from features are correct"""
    for l in range(len(det)):#lsort[:100]:
        scr=det[l]["scr"]
        idm=det[l]["id"]
        r=det[l]["hog"]    
        m1=models[idm]["ww"][0]
        m2=f.hog[r]
        res=det[l]["def"]
        mcost=models[idm]["cost"].astype(numpy.float32)
        dfeat,edge=crf3.getfeat_full(m2,0,res)
        print "Scr",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-models[idm]["rho"],"Error",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr-models[idm]["rho"]
        if numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr-models[idm]["rho"]>0.0001:
            printf("Error %f too big, there is something wrong!!!"%(numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr-models[idm]["rho"]))
            raw_input()

def getfeature(det,N,f,models,trunc=0):
    """ check if score of detections and score from features are correct"""
    lfeat=[];ledge=[]
    for l in range(len(det)):#lsort[:100]:
        scr=det[l]["scr"]
        idm=det[l]["id"]
        r=det[l]["hog"]    
        m1=models[idm]["ww"][0]
        m2=f.hog[r]
        res=det[l]["def"]
        mcost=models[idm]["cost"].astype(numpy.float32)
        dfeat,edge=crf3.getfeat_fullN(m2,N,res,trunc=trunc)
        lfeat.append(dfeat)
        ledge.append(edge)
        #print "Scr",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-models[idm]["rho"],"Error",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr-models[idm]["rho"]
        if numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr-models[idm]["rho"]>0.0001:
            print("Error %f too big, there is something wrong!!!"%(numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr-models[idm]["rho"]))
            raw_input()
    return lfeat,ledge


def boundingbox(det,N):
    """extract the detection boundig box"""
    for l in range(len(det)):
        scl=det[l]["scl"]
        scr=det[l]["scr"]
        idm=det[l]["id"]
        r=det[l]["hog"]    
        #m2=f.hog[r]
        res=det[l]["def"]
        pos=numpy.zeros(res.shape)
        sf=int(8*N/scl)
        for px in range(res.shape[2]):
            for py in range(res.shape[1]):
                #util.box(py*2*hogpix+res[0,py,px]*hogpix,px*2*hogpix+res[1,py,px]*hogpix,py*2*hogpix+res[0,py,px]*hogpix+2*hogpix,px*2*hogpix+res[1,py,px]*hogpix+2*hogpix, col=col[fres.shape[0]-hy-1], lw=2)  
                pos[0,py,px]=(py)*sf+(res[0,py,px]+1)*sf/N
                pos[1,py,px]=(px)*sf+(res[1,py,px]+1)*sf/N
        det[l]["bbox"]=(numpy.min(pos[0]),numpy.min(pos[1]),numpy.max(pos[0])+sf,numpy.max(pos[1])+sf)

def clip(det,imsize):#imsize(y,x)
    """clip the deteciton bboxes to be inside the image"""
    for idl,l in enumerate(det):
        #print l["bbox"],
        y1=max(0,l["bbox"][0])
        x1=max(0,l["bbox"][1])
        y2=min(imsize[0],l["bbox"][2])
        x2=min(imsize[1],l["bbox"][3])
        det[idl]["bbox"]=(y1,x1,y2,x2)
        #print det[idl]["bbox"]

def visualize(det,N,f,img,fig=300,text=""):
    """visualize a detection and the corresponding featues"""
    pl=pylab
    col=['w','r','g','b','y','c','k','y','c','k']
    pl.figure(fig,figsize=(15,5))
    pl.clf()
    pl.subplot(1,3,1)
    pl.imshow(img)
    pl.title(text)
    im=img
    pad=0
    cc=0
    #rcim=numpy.array([])
    for l in range(len(det)):#lsort[:100]:
        scl=det[l]["scl"]
        idm=det[l]["id"]
        r=det[l]["hog"]
        res=det[l]["def"]
        scr=det[l]["scr"]
        numy=det[l]["def"].shape[1]#cfg.fy[idm]
        numx=det[l]["def"].shape[2]#cfg.fx[idm]
        sf=int(8*N/scl)
        #m1=models[idm]["ww"][0]
        m2=f.hog[r]
        #mcost=models[idm]["cost"].astype(numpy.float32)
        if l==0:
           im2=numpy.zeros((im.shape[0]+sf*numy*2,im.shape[1]+sf*numx*2,im.shape[2]),dtype=im.dtype)
           im2[sf*numy:sf*numy+im.shape[0],sf*numx:sf*numx+im.shape[1]]=im
           rcim=numpy.zeros((sf*numy,sf*numx,3),dtype=im.dtype)
        dfeat,edge=crf3.getfeat_fullN(m2,N,res)
        #print "Scr",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost),"Error",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr
        pl.subplot(1,3,1)
        for px in range(res.shape[2]):
            for py in range(res.shape[1]):
                #util.box(py*2*hogpix+res[0,py,px]*hogpix,px*2*hogpix+res[1,py,px]*hogpix,py*2*hogpix+res[0,py,px]*hogpix+2*hogpix,px*2*hogpix+res[1,py,px]*hogpix+2*hogpix, col=col[fres.shape[0]-hy-1], lw=2)  
                impy=(py)*sf+(res[0,py,px]+1)*sf/N
                impx=(px)*sf+(res[1,py,px]+1)*sf/N
                util.box(impy,impx,impy+sf,impx+sf, col=col[cc%10], lw=1.5)  
                if det[l].has_key("bbox"):
                    util.box(det[l]["bbox"][0],det[l]["bbox"][1],det[l]["bbox"][2],det[l]["bbox"][3],col=col[cc%10],lw=2)
                #rcim[sf*py:sf*(py+1),sf*px:sf*(px+1)]=im2[sf*numy+impy:sf*numy+impy+sf,sf*numx+impx:sf*
                #util.box(py*2*hogpix+res[0,py,px]*hogpix,px*2*hogpix+res[1,py,px]*hogpix,py*2*hogpix+res[0,py,px]*hogpix+2*hogpix,px*2*hogpix+res[1,py,px]*hogpix+2*hogpix, col=col[fres.shape[0]-hy-1], lw=2)  
                if l==0:
                    rcim[sf*py:sf*(py+1),sf*px:sf*(px+1)]=im2[sf*numy+impy:sf*numy+impy+sf,sf*numx+impx:sf*numx+impx+sf] 
        cc+=1
        if l==0:
            pl.subplot(1,3,2)
            hdet=drawHOG.drawHOG(dfeat)
            pl.imshow(hdet)
            pl.subplot(1,3,3)
            pl.title("%f"%scr)
            pl.imshow(rcim)    
    pl.subplot(1,3,1)    
    #pl.axis([0,img.shape[1],0,img.shape[0]])
    pl.axis("image")
    pl.draw()
    pl.show()
    #raw_input()

def visualize2(det,N,img,bb=[],text=""):
    """visualize a detection and the corresponding featues"""
    pl=pylab
    col=['w','r','g','b','y','c','k','y','c','k']
    pl.figure(300,figsize=(8,4))
    pl.clf()
    pl.subplot(1,2,1)
    pl.title(text)
    pl.imshow(img)
    im=img
    pad=0
    cc=0
    if bb!=[]:
        util.box(bb[0],bb[1],bb[2],bb[3], col="b--", lw=2)  
    for l in range(len(det)):#lsort[:100]:
        scl=det[l]["scl"]
        idm=det[l]["id"]
        r=det[l]["hog"]
        res=det[l]["def"]
        scr=det[l]["scr"]
        numy=det[l]["def"].shape[1]#cfg.fy[idm]
        numx=det[l]["def"].shape[2]#cfg.fx[idm]
        sf=int(8*N/scl)
        #m2=f.hog[r]
        if l==0:
           im2=numpy.zeros((im.shape[0]+sf*numy*2,im.shape[1]+sf*numx*2,im.shape[2]),dtype=im.dtype)
           im2[sf*numy:sf*numy+im.shape[0],sf*numx:sf*numx+im.shape[1]]=im
           rcim=numpy.zeros((sf*numy,sf*numx,3),dtype=im.dtype)
        #dfeat,edge=crf3.getfeat_full(m2,pad,res)
        pl.subplot(1,2,1)
        for px in range(res.shape[2]):
            for py in range(res.shape[1]):
                impy=(py)*sf+(res[0,py,px]+1)*sf/N
                impx=(px)*sf+(res[1,py,px]+1)*sf/N
                util.box(impy,impx,impy+sf,impx+sf, col=col[cc%10], lw=1.5)  
                if det[l].has_key("bbox"):
                    util.box(det[l]["bbox"][0],det[l]["bbox"][1],det[l]["bbox"][2],det[l]["bbox"][3],col=col[cc%10],lw=2)
                if l==0:
                    rcim[sf*py:sf*(py+1),sf*px:sf*(px+1)]=im2[sf*numy+impy:sf*numy+impy+sf,sf*numx+impx:sf*numx+impx+sf] 
        cc+=1
        if l==0:
            pl.subplot(1,2,2)
            pl.title("scr:%.3f id:%d"%(scr,idm))
            pl.imshow(rcim)    
    pl.subplot(1,2,1)    
    #pl.axis("image")
    pl.axis([0,img.shape[1],img.shape[0],0])
    pl.draw()
    pl.show()


def rundet(img,N,models,numhyp=5,interv=10,aiter=3,restart=0,trunc=0):
    "run the CRF optimization at each level of the HOG pyramid"
    f=pyrHOG2.pyrHOG(img,interv=interv,savedir="",hallucinate=True,cformat=True)
    det=[]
    for idm,m in enumerate(models):
        mcost=m["cost"].astype(numpy.float32)
        m1=m["ww"][0]
        numy=m["ww"][0].shape[0]
        numx=m["ww"][0].shape[1]
        for r in range(len(f.hog)):
            m2=f.hog[r]
            #print numy,numx
            lscr,fres=crf3.match_fullN(m1,m2,N,mcost,show=False,feat=False,rot=False,numhyp=numhyp,aiter=aiter,restart=restart,trunc=trunc)
            #print "Total time",time.time()-t
            #print "Score",lscr
            idraw=False
            if idraw:
                import drawHOG
                #rec=drawHOG.drawHOG(dfeat)
                pylab.figure(figsize=(15,5))
                #subplot(1,2,1)
                #imshow(rec)
                pylab.title("Reconstructed HOG Image (Learned Costs)")
                pylab.subplot(1,2,2)
                img=drawHOG.drawHOG(m2)
            for idt in range(len(lscr)):
                det.append({"id":m["id"],"hog":r,"scl":f.scale[r],"def":fres[idt],"scr":lscr[idt]-models[idm]["rho"]})
    det.sort(key=lambda by: -by["scr"])
    #if cfg.show:
    #    pylab.draw()
    #    pylab.show()
    return [f,det]

import math as mt

def rundetw(img,N,models,numhyp=5,interv=10,aiter=3,restart=0,trunc=0,wstepy=-1,wstepx=-1):
    "run the CRF optimization at each level of the HOG pyramid but in a sliding window way"
    f=pyrHOG2.pyrHOG(img,interv=interv,savedir="",hallucinate=True,cformat=True)
    #add maximum padding to each hog    
    maxfy=max([x["ww"][0].shape[0] for x in models])
    maxfx=max([x["ww"][0].shape[1] for x in models])
    padf=[]
    #add 1 more maxf pad at the end to account for remaining parts of the grid
    for idl,l in enumerate(f.hog):
        #padf.append(numpy.zeros((f.hog[idl].shape[0]+2*maxfy,f.hog[idl].shape[1]+2*maxfx,f.hog[idl].shape[2]),dtype=f.hog[idl].dtype))
        padf.append(numpy.zeros((f.hog[idl].shape[0]+2*maxfy+maxfy,f.hog[idl].shape[1]+2*maxfx+maxfx,f.hog[idl].shape[2]),dtype=f.hog[idl].dtype))
        padf[-1][maxfy:maxfy+f.hog[idl].shape[0],maxfx:maxfx+f.hog[idl].shape[1]]=f.hog[idl]
    det=[]
    for idm,m in enumerate(models):
        #for each model the minimum window size is 2 times the model size
        #minwy=m["ww"].shape[0]
        mcost=m["cost"].astype(numpy.float32)
        m1=m["ww"][0]
        numy=m["ww"][0].shape[0]
        numx=m["ww"][0].shape[1]
        minstepy=max(wstepy,2*numy)
        minstepx=max(wstepx,2*numx)
        if wstepy==-1:
            wstepy=numy+1
        if wstepx==-1:
            wstepx=numx+1
        for r in range(len(padf)):
            #m2=padf[r]
            #print "###############scale %d (%d,%d)#############"%(r,padf[r].shape[0],padf[r].shape[1])
            #scan the image with step wstepx-y
            for wy in range(((padf[r].shape[0]-minstepy)/wstepy)):
                for wx in range(((padf[r].shape[1]-minstepx)/wstepx)):
                    #print "WY:",wy,"WX",wx
                    m2=padf[r][wy*wstepy:wy*wstepy+minstepy,wx*wstepx:wx*wstepx+minstepx]
                    #print m2.shape
                    lscr,fres=crf3.match_fullN_nopad(m1,m2,N,mcost,show=False,feat=False,rot=False,numhyp=numhyp,aiter=aiter,restart=restart,trunc=trunc)
                    for idt in range(len(lscr)):
                        det.append({"id":m["id"],"hog":r,"scl":f.scale[r],"def":(fres[idt].T+numpy.array([wstepy*wy-maxfy,wstepx*wx-maxfx]).T).T,"scr":lscr[idt]-models[idm]["rho"]})
    det.sort(key=lambda by: -by["scr"])
    #if cfg.show:
    #    pylab.draw()
    #    pylab.show()
    return [f,det]




def rundetc(img,N,models,numhyp=5,interv=10,aiter=3,restart=0,trunc=0,bbox=None):
    "run the CRF optimization at each level of the HOG pyramid adding constraints to force the  detection to be in the bounding box"
    f=pyrHOG2.pyrHOG(img,interv=interv,savedir="",hallucinate=True,cformat=True)
    det=[]
    nbbox=None
    for idm,m in enumerate(models):
        mcost=m["cost"].astype(numpy.float32)
        m1=m["ww"][0]
        numy=m["ww"][0].shape[0]
        numx=m["ww"][0].shape[1]
        for r in range(len(f.hog)):
            m2=f.hog[r]
            #print numy,numx
            if bbox!=None:
                nbbox=numpy.array(bbox)*f.scale[r]
            lscr,fres=crf3.match_fullN(m1,m2,N,mcost,show=False,feat=False,rot=False,numhyp=numhyp,aiter=aiter,restart=restart,trunc=trunc,bbox=nbbox)
            #print "Total time",time.time()-t
            #print "Score",lscr
            idraw=False
            if idraw:
                import drawHOG
                #rec=drawHOG.drawHOG(dfeat)
                pylab.figure(figsize=(15,5))
                #subplot(1,2,1)
                #imshow(rec)
                pylab.title("Reconstructed HOG Image (Learned Costs)")
                pylab.subplot(1,2,2)
                img=drawHOG.drawHOG(m2)
            for idt in range(len(lscr)):
                det.append({"id":m["id"],"hog":r,"scl":f.scale[r],"def":fres[idt],"scr":lscr[idt]-models[idm]["rho"]})
    det.sort(key=lambda by: -by["scr"])
    #if cfg.show:
    #    pylab.draw()
    #    pylab.show()
    print "Number of detections:",len(det)
    return [f,det]


def rundetbb(img,N,models,numdet=50,interv=10,aiter=3,restart=0,trunc=0):
    "run the CRF optimization at each level of the HOG pyramid but using branch and bound algorithm"
    #note that branch and bound sometime is generating more than once the same hipothesis
    # I do not know yet why...
    #Maybe the punishment to repeat a location is not high enough
    #print "Branc and bound"
    f=pyrHOG2.pyrHOG(img,interv=interv,savedir="",hallucinate=True,cformat=True)
    ldet2=[]
    for idm,m in enumerate(models):
        mcost=m["cost"].astype(numpy.float32)
        m1=m["ww"][0]
        numy=m["ww"][0].shape[0]
        numx=m["ww"][0].shape[1]
        ldet=crf3.match_bbN(m1,f.hog,N,mcost,show=False,rot=False,numhyp=numdet,aiter=aiter,restart=restart,trunc=trunc)
        for l in ldet:
            r=l["scl"]
            ldet2.append({"id":m["id"],"hog":r,"scl":f.scale[r],"def":l["def"][0],"scr":l["scr"]-models[idm]["rho"]})            
    ldet2.sort(key=lambda by: -by["scr"])
    return [f,ldet2]

def detectCrop(a):
    """
        substitutes the entire image with the corresponding bounding box if there
    """
    i=a["idim"]
    imname=a["file"]
    bbox=a["bbox"]
    models=a["models"]
    cfg=a["cfg"]
    imageflip=a["flip"]
    #if len(a)<=5:
    #    imageflip=False
    #else:
    #    imageflip=a[5]
    img=util.myimread(imname,resize=cfg.resize)
    if imageflip:
        img=util.myimread(imname,True,resize=cfg.resize)
        if bbox!=None:
             bbox = util.flipBBox(img,bbox)
    #maxy=numpy.min([x["ww"][0].shape[0] for x in models])    
    #maxx=numpy.min([x["ww"][0].shape[1] for x in models])    
    if (bbox!=[]):
        marginy=(bbox[2]-bbox[0])*0.5 #safe margin to be sure the border is not used
        marginx=(bbox[3]-bbox[1])*0.5
        y1=max(0,bbox[0]-marginy);x1=max(0,bbox[1]-marginx)
        y2=min(bbox[2]+marginy,img.shape[0]);x2=min(bbox[3]+marginx,img.shape[1])
        img=img[y1:y2,x1:x2]
    res=rundet(img,cfg,models)
    return res
















