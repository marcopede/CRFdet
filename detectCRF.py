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
    #print "Old bbox:",bbox
    extbbox=numpy.array(bbox[:6])
    if bbox[0]<=1 and cfg.useclip:
        extbbox[0]=-bbox[2]
        extbbox[4]=1
    if bbox[1]<=1 and cfg.useclip:
        extbbox[1]=-bbox[3]
        extbbox[4]=1
    if bbox[2]>=img.shape[0]-2 and cfg.useclip:
        extbbox[2]=img.shape[0]+(img.shape[0]-bbox[0])
        extbbox[4]=1
    if bbox[3]>=img.shape[1]-2 and cfg.useclip:
        extbbox[3]=img.shape[1]+(img.shape[1]-bbox[1])
        extbbox[4]=1
    #print "New bbox:",extbbox
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
        #idm=numpy.where(dist<0.5)[0] #
        idm=numpy.where(dist<0.7)[0] #
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
                [f,det]=rundetbb(img,cfg.N,cfg.E,selmodels,numdet=cfg.numhypPOS, interv=cfg.intervPOS,aiter=cfg.aiterPOS,restart=cfg.restartPOS,trunc=cfg.trunc,useFastDP=cfg.useFastDP)
            else:         
                #[f,det]=rundet(img,cfg.N,selmodels,numhyp=cfg.numhypPOS,interv=cfg.intervPOS,aiter=cfg.aiterPOS,restart=cfg.restartPOS,trunc=cfg.trunc)
                [f,det]=rundetc(img,cfg.N,cfg.E,selmodels,numhyp=cfg.numhypPOS,interv=cfg.intervPOS,aiter=cfg.
aiterPOS,restart=cfg.restartPOS,trunc=cfg.trunc,bbox=extnewbbox,useFastDP=cfg.useFastDP)
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
        feat,edge=getfeature([det[best]],cfg.N,cfg.E,f,models,cfg.trunc)
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
        [f,det]=rundetbb(img,cfg.N,cfg.E,models,minthr=-1.0,numdet=cfg.numhypNEG,interv=cfg.intervNEG,aiter=cfg.aiterNEG,restart=cfg.restartNEG,trunc=cfg.trunc,useFastDP=cfg.useFastDP)
    else:
        [f,det]=rundet(img,cfg.N,cfg.E,models,numhyp=cfg.numhypNEG,interv=cfg.intervNEG,aiter=cfg.aiterNEG,restart=cfg.restartNEG,trunc=cfg.trunc)
    ####
    #for idl,l in enumerate(det[1:]):
    #    if abs(det[idl]["scr"]-l["scr"])<0.00000001:
    #        print "Two same detection:"
    #        print det[idl]["scr"],det[idl]["scr"]
    #        raw_input()
    #####
    ldet=[]
    lfeat=[]
    ledge=[]
    for idl,l in enumerate(det[:cfg.numneg]):
        #add bias
        #det[idl]["scr"]-=models[det[idl]["id"]]["rho"]/float(cfg.bias)
        if det[idl]["scr"]>-1:
            det[idl]["idim"]=el["file"].split("/")[-1]
            ldet.append(det[idl])
            feat,edge=getfeature([det[idl]],cfg.N,cfg.E,f,models,cfg.trunc)
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
        [f,det]=rundetbb(img,cfg.N,cfg.E,models,minthr=-1.0,numdet=cfg.numhypNEG,interv=cfg.intervNEG,aiter=cfg.aiterNEG,restart=cfg.restartNEG,trunc=cfg.trunc,useFastDP=cfg.useFastDP)
    else:
        [f,det]=rundet(img,cfg.N,cfg.E,models,numhyp=cfg.numhypNEG,interv=cfg.intervNEG,aiter=cfg.aiterNEG,restart=cfg.restartNEG,trunc=cfg.trunc)
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
                #if ovr>0.3 and (cfg.db!="inria" or ovr<0):#is not a false positive 
                if ovr>0.3:
                    skip=True
                if cfg.db=="inria" and ovr<0.1:
                    skip=True
            if not(skip):
                det[idl]["idim"]=el["file"].split("/")[-1]
                ldet.append(det[idl])
                feat,edge=getfeature([det[idl]],cfg.N,cfg.E,f,models,cfg.trunc)
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
    minthr=-2
    models=el["models"]
    if models[0].has_key("thr"):
        minthr="Learned"
    #[f,det]=detectCRF.detectCrop(el)
    print "----Image-%s-(%d)-----------"%(el["file"].split("/")[-1],el["idim"])
    imname=el["file"]
    bbox=el["bbox"]
    cfg=el["cfg"]
    imageflip=el["flip"]
    if imageflip:
        img=util.myimread(imname,True,resize=cfg.resize)
    else:
        img=util.myimread(imname,resize=cfg.resize)
    #imageflip=el["flip"]
    if cfg.usebbTEST:
        if cfg.useswTEST:
            [f,det]=rundetwbb(img,cfg.N,models,numdet=cfg.numhypTEST*len(models),interv=cfg.intervTEST,aiter=cfg.aiterTEST,restart=cfg.restartTEST,trunc=cfg.trunc,wstepy=cfg.swstepy,wstepx=cfg.swstepx)
        else:
            [f,det]=rundetbb(img,cfg.N,cfg.E,models,minthr=minthr,numdet=cfg.numhypTEST,interv=cfg.intervTEST,aiter=cfg.aiterTEST,restart=cfg.restartTEST,trunc=cfg.trunc,useFastDP=cfg.useFastDP)
    else:
        if cfg.useswTEST:
            [f,det]=rundetw2(img,cfg.N,models,numhyp=cfg.numhypTEST,interv=cfg.intervTEST,aiter=cfg.aiterTEST,restart=cfg.restartTEST,trunc=cfg.trunc,
            wstepy=cfg.swstepy,wstepx=cfg.swstepx,wsizey=cfg.swsizey,wsizex=cfg.swsizex,forcestep=cfg.swforcestep)
        else:
            [f,det]=rundet(img,cfg.N,cfg.E,models,numhyp=cfg.numhypTEST,interv=cfg.intervTEST,aiter=cfg.aiterTEST,restart=cfg.restartTEST,trunc=cfg.trunc)
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
        if abs(numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr-models[idm]["rho"])>0.0001:
            printf("Error %f too big, there is something wrong!!!"%(numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr-models[idm]["rho"]))
            raw_input()

def getfeature(det,N,E,f,models,trunc=0):
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
        dfeat,edge=crf3.getfeat_fullN(m2,N,E,res,trunc=trunc)
        lfeat.append(dfeat)
        ledge.append(edge)
        #print "Scr",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-models[idm]["rho"],"Error",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr-models[idm]["rho"]
        if abs(numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr-models[idm]["rho"])>0.0001:
            print("Error %f too big, there is something wrong!!!"%(numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr-models[idm]["rho"]))
            print "Component:",det[l]["id"]
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
        sh=8/float(scl)
        sf=(8*N/float(scl))
        #print "Size hog",sh,"Size part",sf
        for px in range(res.shape[2]):
            for py in range(res.shape[1]):
                #util.box(py*2*hogpix+res[0,py,px]*hogpix,px*2*hogpix+res[1,py,px]*hogpix,py*2*hogpix+res[0,py,px]*hogpix+2*hogpix,px*2*hogpix+res[1,py,px]*hogpix+2*hogpix, col=col[fres.shape[0]-hy-1], lw=2)  
                pos[0,py,px]=int((py)*sf)+int((res[0,py,px]+1)*sh)
                pos[1,py,px]=int((px)*sf)+int((res[1,py,px]+1)*sh)
        det[l]["bbox"]=(numpy.min(pos[0]),numpy.min(pos[1]),numpy.max(pos[0])+int(sf),numpy.max(pos[1])+int(sf))

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

def visualize2(det,N,img,bb=[],text="",color=None,line=False):
    """visualize a detection and the corresponding featues"""
    pl=pylab
    if color!=None:
        col=color
    else:
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
        sf=float(8*N/scl)
        #m2=f.hog[r]
        if l==0:
           #im2=numpy.zeros((im.shape[0]+sf*numy*2,im.shape[1]+sf*numx*2,im.shape[2]),dtype=im.dtype)
            #for sliding windows to work
           im2=numpy.zeros((im.shape[0]+sf*numy*4,im.shape[1]+sf*numx*4,im.shape[2]),dtype=im.dtype)
           im2[sf*2*numy:sf*2*numy+im.shape[0],sf*2*numx:sf*2*numx+im.shape[1]]=im
           rcim=numpy.zeros((sf*numy+1,sf*numx+1,3),dtype=im.dtype)
        #dfeat,edge=crf3.getfeat_full(m2,pad,res)
        pl.subplot(1,2,1)
        for px in range(res.shape[2]):
            for py in range(res.shape[1]):
                impy=int((py)*sf+(res[0,py,px]+1)*sf/N)
                impx=int((px)*sf+(res[1,py,px]+1)*sf/N)
                if line:
                    if py<res.shape[1]-1: #vertical
                        impy2=int((py+1)*sf+(res[0,py+1,px]+1)*sf/N)
                        impx2=int((px)*sf+(res[1,py+1,px]+1)*sf/N)
                        dst=((impx-impx2)/sf)**2+((impy-impy2)/sf)**2
                        #print dst
                        pylab.plot([impx+sf/2.0,impx2+sf/2.0],[impy+sf/2.0,impy2+sf/2.0],col[cc%10]+'.-',markersize=10.0,lw=5/(float(dst)+1))
                    if px<res.shape[2]-1: #horizontal
                        impy2=int((py)*sf+(res[0,py,px+1]+1)*sf/N)
                        impx2=int((px+1)*sf+(res[1,py,px+1]+1)*sf/N)
                        dst=((impx-impx2)/sf)**2+((impy-impy2)/sf)**2
                        pylab.plot([impx+sf/2.0,impx2+sf/2.0],[impy+sf/2.0,impy2+sf/2.0],col[cc%10]+'.-',markersize=10.0,lw=5/(float(dst)+1))
                else:
                    util.box(impy,impx,impy+int(sf),impx+int(sf), col=col[cc%10], lw=1.5)  
                if det[l].has_key("bbox"):
                    util.box(det[l]["bbox"][0],det[l]["bbox"][1],det[l]["bbox"][2],det[l]["bbox"][3],col=col[cc%10],lw=2)
                if l==0:
                    #if int(sf*numy)+impy>im2.shape[0] or int(sf*numx)+impx>im2.shape[1]:
                    #    rcim[int(sf*py):int(sf*(py+1)),int(sf*px):int(sf*(px+1))]=0
                    #else:
                        rcim[int(sf*py):int(sf*py)+int(sf)+1,
int(sf*px):int(sf*px)+int(sf)+1]=im2[int(2*sf*numy)+impy:int(2*sf*numy)+impy+int(sf)+1,int(2*sf*numx)+impx:int(2*sf*numx)+impx+int(sf)+1] 
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


def visualizeDet(det,N,img,bb=[],text="",color=None,line=False):
    """visualize a detection and the corresponding featues"""
    pl=pylab
    if color!=None:
        col=color
    else:
        col=['w','r','g','b','y','c','k','y','c','k']
    im=img
    pad=0
    cc=0
    pl.imshow(img)
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
        sf=float(8*N/scl)
        #m2=f.hog[r]
        if l==0:
           #im2=numpy.zeros((im.shape[0]+sf*numy*2,im.shape[1]+sf*numx*2,im.shape[2]),dtype=im.dtype)
            #for sliding windows to work
           im2=numpy.zeros((im.shape[0]+sf*numy*2,im.shape[1]+sf*numx*2,im.shape[2]),dtype=im.dtype)
           im2[sf*numy:sf*numy+im.shape[0],sf*numx:sf*numx+im.shape[1]]=im
           rcim=numpy.zeros((sf*numy+1,sf*numx+1,3),dtype=im.dtype)
        #dfeat,edge=crf3.getfeat_full(m2,pad,res)
        for px in range(res.shape[2]):
            for py in range(res.shape[1]):
                impy=int((py)*sf+(res[0,py,px]+1)*sf/N)
                impx=int((px)*sf+(res[1,py,px]+1)*sf/N)
                if line:
                    if py<res.shape[1]-1: #vertical
                        impy2=int((py+1)*sf+(res[0,py+1,px]+1)*sf/N)
                        impx2=int((px)*sf+(res[1,py+1,px]+1)*sf/N)
                        dst=((impx-impx2)/sf)**2+((impy-impy2)/sf)**2
                        #print dst
                        pylab.plot([impx+sf/2.0,impx2+sf/2.0],[impy+sf/2.0,impy2+sf/2.0],col[cc%10]+'.-',markersize=10.0,lw=5/(float(dst)+1))
                    if px<res.shape[2]-1: #horizontal
                        impy2=int((py)*sf+(res[0,py,px+1]+1)*sf/N)
                        impx2=int((px+1)*sf+(res[1,py,px+1]+1)*sf/N)
                        dst=((impx-impx2)/sf)**2+((impy-impy2)/sf)**2
                        pylab.plot([impx+sf/2.0,impx2+sf/2.0],[impy+sf/2.0,impy2+sf/2.0],col[cc%10]+'.-',markersize=10.0,lw=5/(float(dst)+1))
                else:
                    util.box(impy,impx,impy+int(sf),impx+int(sf), col=col[cc%10], lw=1.5)  
                if det[l].has_key("bbox"):
                    util.box(det[l]["bbox"][0],det[l]["bbox"][1],det[l]["bbox"][2],det[l]["bbox"][3],col=col[cc%10],lw=2)
                if l==0:
                    rcim[int(sf*py):int(sf*py)+int(sf)+1,
int(sf*px):int(sf*px)+int(sf)+1]=im2[int(sf*numy)+impy:int(sf*numy)+impy+int(sf)+1,int(sf*numx)+impx:int(sf*numx)+impx+int(sf)+1] 
        cc+=1
    #pl.axis("image")
    pl.axis([0,img.shape[1],img.shape[0],0])
    pl.draw()
    pl.show()

def visualizeRec(det,N,img,bb=[],text="",color=None,line=False):
    """visualize a detection and the corresponding featues"""
    pl=pylab
    if color!=None:
        col=color
    else:
        col=['w','r','g','b','y','c','k','y','c','k']
    #pl.figure(300,figsize=(8,4))
    #pl.clf()
    #pl.subplot(1,2,1)
    #pl.title(text)
    #pl.imshow(img)
    im=img
    pad=0
    cc=0
    for l in range(len(det)):#lsort[:100]:
        scl=det[l]["scl"]
        idm=det[l]["id"]
        r=det[l]["hog"]
        res=det[l]["def"]
        scr=det[l]["scr"]
        numy=det[l]["def"].shape[1]#cfg.fy[idm]
        numx=det[l]["def"].shape[2]#cfg.fx[idm]
        sf=float(8*N/scl)
        #m2=f.hog[r]
        if l==0:
           #im2=numpy.zeros((im.shape[0]+sf*numy*2,im.shape[1]+sf*numx*2,im.shape[2]),dtype=im.dtype)
            #for sliding windows to work
           im2=numpy.zeros((im.shape[0]+sf*numy*2+1,im.shape[1]+sf*numx*2+1,im.shape[2]),dtype=numpy.float32)
           im2[sf*numy:sf*numy+im.shape[0],sf*numx:sf*numx+im.shape[1]]=im
           rcim=numpy.zeros((sf*(numy+1)+1,sf*(numx+1)+1,3),dtype=numpy.float32)
           mask=numpy.zeros((int(sf)+int(sf),int(sf)+int(sf),3),dtype=numpy.float32)
           temp=(mask[0:mask.shape[0]/2].T+numpy.linspace(0,0.5,mask.shape[0]/2)).T
           mask[0:mask.shape[0]/2]=temp
           #pl.figure();pl.imshow(mask);pl.show()
           #raw_input()
           ll=mask.shape[0]-mask.shape[0]/2
           aux=numpy.zeros((3,mask.shape[1]/2))
           mask[mask.shape[0]/2:]=(mask[mask.shape[0]/2:].T+numpy.linspace(0.5,0,ll).T).T
           #pl.figure();pl.imshow(mask);pl.show()
           #raw_input()
           aux[:]=numpy.linspace(0,0.5,mask.shape[1]/2)
           mask[:,0:mask.shape[1]/2]=mask[:,0:mask.shape[1]/2]+aux.T
           ll=mask.shape[1]-mask.shape[1]/2
           aux=numpy.zeros((3,ll))
           aux[:]=numpy.linspace(0.5,0,ll)
           mask[:,ll:]=mask[:,ll:]+aux.T[:]
           #pl.figure();pl.imshow(mask);pl.show()
           #raw_input()
        #dfeat,edge=crf3.getfeat_full(m2,pad,res)
        #pl.subplot(1,2,1)
        for px in range(res.shape[2]):
            for py in range(res.shape[1]):
                impy=int((py)*sf+(res[0,py,px]+1)*sf/N)
                impx=int((px)*sf+(res[1,py,px]+1)*sf/N)
                #rcim[int(sf*py):int(sf*py)+int(sf)+int(sf)+1,int(sf*px):int(sf*px)+int(sf)+int(sf)+1]=rcim[int(sf*py):int(sf*py)+int(sf)+int(sf)+1,int(sf*px):int(sf*px)+int(sf)+int(sf)+1]+mask*im2[int(sf*numy)+int(-sf/2)+impy:int(sf*numy)+int(-sf/2)+impy+int(sf)+int(sf)+1,int(sf*numx)+int(-sf/2)+impx:int(sf*numx)+int(-sf/2)+impx+int(sf)+int(sf)+1] 
                rcim[int(sf)*py:int(sf)*py+int(sf)+int(sf),int(sf)*px:int(sf)*px+int(sf)+int(sf)]=rcim[int(sf)*py:int(sf)*py+int(sf)+int(sf),int(sf)*px:int(sf)*px+int(sf)+int(sf)]+mask*im2[int(sf)*numy+int(-sf/2)+impy:int(sf)*numy+int(-sf/2)+impy+int(sf)+int(sf),int(sf)*numx+int(-sf/2)+impx:int(sf)*numx+int(-sf/2)+impx+int(sf)+int(sf)] 
        cc+=1
        if l==0:
            #pl.subplot(1,2,2)
            #pl.title("scr:%.3f id:%d"%(scr,idm))
            pl.imshow(rcim[sf/2:-sf/2,sf/2:-sf/2]/rcim.max())#.astype(numpy.uint8))    


def visualizeRec2(det,N,img,bb=[],text="",color=None,line=False):
    """visualize a detection and the corresponding featues"""
    pl=pylab
    if color!=None:
        col=color
    else:
        col=['w','r','g','b','y','c','k','y','c','k']
    #pl.figure(300,figsize=(8,4))
    #pl.clf()
    #pl.subplot(1,2,1)
    #pl.title(text)
    #pl.imshow(img)
    im=img
    pad=0
    cc=0
    for l in range(len(det)):#lsort[:100]:
        scl=det[l]["scl"]
        idm=det[l]["id"]
        r=det[l]["hog"]
        res=det[l]["def"]
        #res=numpy.ones((res.shape[0],res.shape[1],res.shape[2]))
        scr=det[l]["scr"]
        numy=det[l]["def"].shape[1]#cfg.fy[idm]
        numx=det[l]["def"].shape[2]#cfg.fx[idm]
        sf=float(8*N/scl)
        imres=numpy.zeros((res.shape[0],res.shape[1]+2,res.shape[2]+2))
        from scipy.interpolate import griddata,interp2d,LinearNDInterpolator
        from scipy.ndimage.interpolation import map_coordinates
        for px in range(res.shape[2]):
            for py in range(res.shape[1]):
                imres[0,py+1,px+1]=int((py)*sf+(res[0,py,px]+1)*sf/N+sf)
                imres[1,py+1,px+1]=int((px)*sf+(res[1,py,px]+1)*sf/N+sf)
        imres[0,0,:]=imres[0,1,:]-sf;imres[1,0,:]=imres[1,1,:]
        imres[0,-1,:]=imres[0,-2,:]+sf;imres[1,-1,:]=imres[1,-2,:]
        imres[1,:,0]=imres[1,:,1]-sf;imres[0,:,0]=imres[0,:,1]
        imres[1,:,-1]=imres[1,:,-2]+sf;imres[0,:,-1]=imres[0,:,-2]
        #imres[0,0,0]=imres[0,1,1]-sf;imres[1,0,0]=imres[1,1,1]-sf
        #imres[0,-1,0]=imres[0,-2,1]+sf;imres[0,-1,:]=imres[0,-2,:]+sf
        #imres[1,:,0]=imres[1,:,1]-sf;imres[0,:,0]=imres[0,:,1]-sf
        #imres[1,:,-1]=imres[1,:,-2]+sf;imres[0,:,-1]=imres[0,:,-2]+sf
        grid = numpy.mgrid[0:numy+2,0:numx+2]
        gridf = numpy.mgrid[0:numy:1j*sf*numy, 0:numx:1j*sf*numx]
        #grid2=numpy.concatenate((grid[0].reshape((-1,1)),grid[1].flatten().reshape((-1,1))),1)
        meshy=griddata((grid[0].flatten(),grid[1].flatten()),imres[0,grid[0].flatten(),grid[1].flatten()],(gridf[0],gridf[1]), method='linear')
        meshx=griddata((grid[0].flatten(),grid[1].flatten()),imres[1,grid[0].flatten(),grid[1].flatten()],(gridf[0],gridf[1]), method='linear')
        rec0=map_coordinates(im[:,:,0], numpy.array([meshy,meshx]), order=1, mode='constant')
        rec1=map_coordinates(im[:,:,1], numpy.array([meshy,meshx]), order=1, mode='constant')
        rec2=map_coordinates(im[:,:,2], numpy.array([meshy,meshx]), order=1, mode='constant')
        cc+=1
        if l==0:
            #pl.subplot(1,2,2)
            #pl.title("scr:%.3f id:%d"%(scr,idm))
            rec=numpy.zeros((rec0.shape[0],rec0.shape[1],3))
            rec[:,:,0]=rec0;rec[:,:,1]=rec1;rec[:,:,2]=rec2
            pl.imshow(rec/rec.max())#.astype(numpy.uint8))
            pl.axis([0,rec.shape[1],rec.shape[0],0])    
            pl.draw()
            pl.show()
            


def rundet(img,N,E,models,numhyp=5,interv=10,aiter=3,restart=0,trunc=0,sort=True):
    "run the CRF optimization at each level of the HOG pyramid"
    f=pyrHOG2.pyrHOG(img,interv=interv,savedir="",hallucinate=True,cformat=True)
    det=[]
    for idm,m in enumerate(models):
        mcost=m["cost"].astype(numpy.float32)
        m1=m["ww"][0]
        #numy=m["ww"][0].shape[0]
        #numx=m["ww"][0].shape[1]
        for r in range(len(f.hog)):
            m2=f.hog[r]
            #print numy,numx
            lscr,fres=crf3.match_fullN(m1,m2,N,E,mcost,show=False,feat=False,rot=False,numhyp=numhyp,aiter=aiter,restart=restart,trunc=trunc)
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
    if sort:
        det.sort(key=lambda by: -by["scr"])
    return [f,det]

import math as mt

def rundetw(img,N,models,numhyp=5,interv=10,aiter=3,restart=0,trunc=0,wstepy=-1,wstepx=-1,wsizey=-1,wsizex=-1,forcestep=False,sort=True):
    "run the CRF optimization at each level of the HOG pyramid but in a sliding window way"
    count=0
    #forcestep=True --> step=min(step,size-num*2)
    #forcestep=False --> forcesize size=max(size,step+num*2)
    f=pyrHOG2.pyrHOG(img,interv=interv,savedir="",hallucinate=True,cformat=True)
    #add maximum padding to each hog    
    maxfy=max([x["ww"][0].shape[0] for x in models])
    maxfx=max([x["ww"][0].shape[1] for x in models])
    padf=[]
    #add 1 more maxf pad at the end to account for remaining parts of the grid
    for idl,l in enumerate(f.hog):
        #padf.append(numpy.zeros((f.hog[idl].shape[0]+2*maxfy,f.hog[idl].shape[1]+2*maxfx,f.hog[idl].shape[2]),dtype=f.hog[idl].dtype))
        padf.append(numpy.zeros((f.hog[idl].shape[0]+2*maxfy+2*maxfy,f.hog[idl].shape[1]+2*maxfx+2*maxfx,f.hog[idl].shape[2]),dtype=f.hog[idl].dtype))
        padf[-1][maxfy:maxfy+f.hog[idl].shape[0],maxfx:maxfx+f.hog[idl].shape[1]]=f.hog[idl]
    det=[]
    iwstepy=wstepy
    iwstepx=wstepx
    iwsizey=wsizey
    iwsizex=wsizex
    for idm,m in enumerate(models):
        #for each model the minimum window size is 2 times the model size
        #minwy=m["ww"].shape[0]
        mcost=m["cost"].astype(numpy.float32)
        m1=m["ww"][0]
        numy=m["ww"][0].shape[0]
        numx=m["ww"][0].shape[1]
        #minimum condition wsize=step+2*num
        #so max def = 2*num
        #if I choose step --> wsize>=step+2*num
        #if I choose wsize --> step<wsize-2*num
        if iwstepy==-1:
            wstepy=2*numy#numy+1
        if iwstepx==-1:
            wstepx=2*numx#numx+1
        if iwsizey<0:
            wsizey=-iwsizey*4*numy
        if iwsizex<0:
            wsizex=-iwsizex*4*numx
        if forcestep:
            wstepy=min(wstepy,wsizey-2*numy)
            wstepx=min(wstepx,wsizex-2*numx)
        else: #forcesize
            wsizey=max(wsizey,wstepy+2*numy)
            wsizex=max(wsizex,wstepx+2*numx)   
        #print "Object Model",numy,numx
        #print "Sliding steps",wstepy,wstepx
        #print "Windows size",wsizey,wsizex
        for r in range(len(padf)):
            #scan the image with step wstepx-y and window size wsizex-y
            for wy in range(((padf[r].shape[0]-(wsizey-wstepy))/wstepy)):
                for wx in range(((padf[r].shape[1]-(wsizex-wstepx))/wstepx)):
                    #print "WY:",wy,"WX",wx
                    m2=padf[r][wy*wstepy:wy*wstepy+wsizey,wx*wstepx:wx*wstepx+wsizex]
                    #print m2.shape
                    count+=1
                    lscr,fres=crf3.match_fullN_nopad(m1,m2,N,mcost,show=False,feat=False,rot=False,numhyp=numhyp,aiter=aiter,restart=restart,trunc=trunc)
                    for idt in range(len(lscr)):
                        det.append({"id":m["id"],"hog":r,"scl":f.scale[r],"def":(fres[idt].T+numpy.array([wstepy*wy-maxfy,wstepx*wx-maxfx]).T).T,"scr":lscr[idt]-models[idm]["rho"]})
    if sort:
        det.sort(key=lambda by: -by["scr"])
    #if cfg.show:
    #    pylab.draw()
    #    pylab.show()
    print "Number evaluations SW:",count
    return [f,det]


def rundetw2(img,N,models,numhyp=5,interv=10,aiter=3,restart=0,trunc=0,wstepy=-1,wstepx=-1,wsizey=-1,wsizex=-1,forcestep=False):
    "run the CRF optimization at each level of the HOG pyramid but in a sliding window way"
    #forcestep=True --> step=min(step,size-num*2)
    #forcestep=False --> forcesize size=max(size,step+num*2)
    f=pyrHOG2.pyrHOG(img,interv=interv,savedir="",hallucinate=True,cformat=True)
    #add maximum padding to each hog    
    maxfy=max([x["ww"][0].shape[0] for x in models])
    maxfx=max([x["ww"][0].shape[1] for x in models])
    padf=[]
    #add 1 more maxf pad at the end to account for remaining parts of the grid
    for idl,l in enumerate(f.hog):
        #padf.append(numpy.zeros((f.hog[idl].shape[0]+2*maxfy,f.hog[idl].shape[1]+2*maxfx,f.hog[idl].shape[2]),dtype=f.hog[idl].dtype))
        padf.append(numpy.zeros((f.hog[idl].shape[0]+2*maxfy+2*maxfy,f.hog[idl].shape[1]+2*maxfx+2*maxfx,f.hog[idl].shape[2]),dtype=f.hog[idl].dtype))
        padf[-1][maxfy:maxfy+f.hog[idl].shape[0],maxfx:maxfx+f.hog[idl].shape[1]]=f.hog[idl]
    det=[]
    iwstepy=wstepy
    iwstepx=wstepx
    iwsizey=wsizey
    iwsizex=wsizex
    for idm,m in enumerate(models):
        #for each model the minimum window size is 2 times the model size
        #minwy=m["ww"].shape[0]
        mcost=m["cost"].astype(numpy.float32)
        m1=m["ww"][0]
        numy=m["ww"][0].shape[0]
        numx=m["ww"][0].shape[1]
        #minimum condition wsize=step+2*num
        #so max def = 2*num
        #if I choose step --> wsize>=step+2*num
        #if I choose wsize --> step<wsize-2*num
        if iwstepy==-1:
            wstepy=2*numy#numy+1
        if iwstepx==-1:
            wstepx=2*numx#numx+1
        if iwsizey<0:
            wsizey=-iwsizey*4*numy
        if iwsizex<0:
            wsizex=-iwsizex*4*numx
        if forcestep:
            wstepy=min(wstepy,wsizey-2*numy)
            wstepx=min(wstepx,wsizex-2*numx)
        else: #forcesize
            wsizey=max(wsizey,wstepy+2*numy)
            wsizex=max(wsizex,wstepx+2*numx)   
        #print "Object Model",numy,numx
        #print "Sliding steps",wstepy,wstepx
        #print "Windows size",wsizey,wsizex
        for r in range(len(padf)):
            #scan the image with step wstepx-y and window size wsizex-y
            for wy in range(max(((padf[r].shape[0]-wsizey-2*maxfy)/wstepy)+2,1)):#at least 1 it
                for wx in range(max(((padf[r].shape[1]-wsizex-2*maxfx)/wstepx)+2,1)):
                    #print "WY:",wy,"WX",wx
                    m2=padf[r][wy*wstepy:wy*wstepy+wsizey,wx*wstepx:wx*wstepx+wsizex]
                    #print m2.shape
                    lscr,fres=crf3.match_fullN_nopad(m1,m2,N,mcost,show=False,feat=False,rot=False,numhyp=numhyp,aiter=aiter,restart=restart,trunc=trunc)
                    for idt in range(len(lscr)):
                        det.append({"id":m["id"],"hog":r,"scl":f.scale[r],"def":(fres[idt].T+numpy.array([wstepy*wy-maxfy,wstepx*wx-maxfx]).T).T,"scr":lscr[idt]-models[idm]["rho"]})

                
    det.sort(key=lambda by: -by["scr"])
    #if cfg.show:
    #    pylab.draw()
    #    pylab.show()
    return [f,det]


def rundetwbb(img,N,models,numdet=5,interv=10,aiter=3,restart=0,trunc=0,wstepy=-1,wstepx=-1,wsizey=-1,wsizex=-1,forcestep=False):
    "run the CRF optimization at each level of the HOG pyramid but in a sliding window way"
    f=pyrHOG2.pyrHOG(img,interv=interv,savedir="",hallucinate=True,cformat=True)
    #add maximum padding to each hog    
    maxfy=max([x["ww"][0].shape[0] for x in models])
    maxfx=max([x["ww"][0].shape[1] for x in models])
    padf=[]
    #add 1 more maxf pad at the end to account for remaining parts of the grid
    for idl,l in enumerate(f.hog):
        #padf.append(numpy.zeros((f.hog[idl].shape[0]+2*maxfy,f.hog[idl].shape[1]+2*maxfx,f.hog[idl].shape[2]),dtype=f.hog[idl].dtype))
        padf.append(numpy.zeros((f.hog[idl].shape[0]+2*maxfy+2*maxfy,f.hog[idl].shape[1]+2*maxfx+2*maxfx,f.hog[idl].shape[2]),dtype=f.hog[idl].dtype))
        padf[-1][maxfy:maxfy+f.hog[idl].shape[0],maxfx:maxfx+f.hog[idl].shape[1]]=f.hog[idl]
    #compute filters and max bounds
    loc=[]
    iwstepy=wstepy
    iwstepx=wstepx
    iwsizey=wsizey
    iwsizex=wsizex
    for idm,m in enumerate(models):
        mcost=m["cost"].astype(numpy.float32)
        m1=m["ww"][0]
        numy=m["ww"][0].shape[0]
        numx=m["ww"][0].shape[1]
        numly=m["ww"][0].shape[0]/N
        numlx=m["ww"][0].shape[1]/N
        if iwstepy==-1:
            #wstepy=numy+1
            wstepy=2*numy#/2
        if iwstepx==-1:
            #wstepx=numx+1
            wstepx=2*numx#/2
        if iwsizey==-1:
            wsizey=-iwsizey*4*numy
        if iwsizex==-1:
            wsizex=-iwsizex*4*numx
        if forcestep:
            wstepy=min(wstepy,wsizey-2*numy)
            wstepx=min(wstepx,wsizex-2*numx)
        else: #forcesize
            wsizey=max(wsizey,wstepy+2*numy)
            wsizex=max(wsizex,wstepx+2*numx)  
        #minstepy=max(iwstepy,2*numy+wstepy)
        #minstepx=max(iwstepx,2*numx+wstepx)
        for r in range(len(padf)):
            #scan the image with step wstepx-y
            for wy in range(((padf[r].shape[0]-(wsizey-wstepy))/wstepy)):
                for wx in range(((padf[r].shape[1]-(wsizex-wstepx))/wstepx)):
                    m2=padf[r][wy*wstepy:wy*wstepy+wsizey,wx*wstepx:wx*wstepx+wsizex]
                    rdata,dmin=crf3.filtering_nopad(m1,m2,N,mcost,trunc=trunc)
                    bound=numpy.sum(rdata.reshape((numly*numlx,-1)).max(1))-dmin*numly*numlx-m["rho"]
                    #id position scale uniquely define a detection
                    loc.append({"id":m["id"],"hog":r,"pos":(wy,wx),"scl":f.scale[r],"bscr":bound,"rdata":rdata,"dmin":dmin,"tied":False})
    #get the best detections
    det=[]                
    nmaxloc=numpy.argmax([x["bscr"] for x in loc])
    while (True):
        #print loc[nmaxloc]["bscr"],
        maxloc=nmaxloc
        #maxloc=numpy.argmax([x["scr"] for x in loc])
        idm=loc[maxloc]["id"]
        dmin=loc[maxloc]["dmin"]
        rdata=loc[maxloc]["rdata"]
        mcost=models[idm]["cost"]#.astype(numpy.float32)
        #dmin=loc[maxloc]["dmin"]
        #print "Best",maxloc,"Dmin",dmin,"Rdata.min()",rdata.min()
        lscr,fres=crf3.matching_nopad(N,mcost,rdata,dmin,numhyp=1,aiter=aiter,restart=restart,trunc=trunc)
        #print "After matching"
        loc[maxloc]["bscr"]=lscr[0]-models[idm]["rho"]#-dmin*numy*numx
        loc[maxloc]["def"]=fres[0]#.copy()#copy not really necessary
        #print "Computed",lscr[0]-models[idm]["rho"]
        loc[maxloc]["tied"]=True
        nmaxloc=numpy.argmax([x["bscr"] for x in loc])
        if loc[nmaxloc]["tied"]:#maxloc==nmaxloc or loc[nmaxloc]["tied"]: #found a solution
            #maxloc=loc[maxloc]["scr"]
            #save solution
            sol=loc[nmaxloc]
            r=sol["hog"]
            dmin=sol["dmin"]
            wy,wx=sol["pos"] 
            idm=sol["id"] 
            rdata=sol["rdata"]
            numy=models[idm]["ww"][0].shape[0]#/N
            numx=models[idm]["ww"][0].shape[1]#/N      
            res2=sol["def"]
            if iwstepy==-1:
                wstepy=2*numy#/2
            if iwstepx==-1:
                wstepx=2*numx#/2
            #minstepy=max(iwstepy,wstepy+2*numy)
            #minstepx=max(iwstepx,wstepx+2*numx)        
            det.append({"id":sol["id"],"hog":sol["hog"],"scl":sol["scl"],"def":(res2.T+numpy.array([wstepy*wy-maxfy,wstepx*wx-maxfx]).T).T,"scr":sol["bscr"]})
            #print "Solution",sol["bscr"]
            #add penalties to forbid same solution
            #res2=res#fres[0]
            numly=rdata.shape[0]
            numlx=rdata.shape[1]
            movy=rdata.shape[2]
            movx=rdata.shape[3]        
            for py in range(res2.shape[1]):
                for px in range(res2.shape[2]):
                    rcy=res2[0,py,px]#+m1.shape[0]
                    rcx=res2[1,py,px]#+m1.shape[1]
                    #data.reshape((data.shape[0],data.shape[1],movy,movx))[py,px,rcy,rcx]=1
                    #rdata.reshape((numy,numx,movy,movx))[py,px,max(0,rcy-1):rcy+2,max(0,rcx-1):rcx+2]=-10
                    rdata[py,px,max(0,rcy-1):rcy+2,max(0,rcx-1):rcx+2]=10
            if len(det)>=numdet:#found all solutions
                break
            #NOT NECESSARY BOUND CAN BE PREVIOUS DETECTION
            #loc[nmaxloc]["bscr"]=numpy.sum(rdata.reshape((numly*numlx,movy*movx)).max(1))-dmin*numly*numlx-models[sol["id"]]["rho"]
            loc[nmaxloc]["tied"]=False
            nmaxloc=numpy.argmax([x["bscr"] for x in loc])
    det.sort(key=lambda by: -by["scr"])
    return [f,det]


def rundetc(img,N,E,models,numhyp=5,interv=10,minsize=-1,aiter=3,restart=0,trunc=0,bbox=None,useFastDP=False):
    "run the CRF optimization at each level of the HOG pyramid adding constraints to force the  detection to be in the bounding box"
    f=pyrHOG2.pyrHOG(img,interv=interv,savedir="",hallucinate=True,cformat=True)
    det=[]
    skip=True
    nbbox=None
    if minsize!=-1:
        f.hog=f.hog[interv/2:interv/2+interv*minsize]#number of octaves to really evaluate
        f.scale=f.scale[interv/2:interv/2+interv*minsize]
    smallpyr=f.hog[interv-2:interv+interv-2]
    for idm,m in enumerate(models):
        pyr=f.hog
        fscl=f.scale
        if m.has_key("big"):
            if m["big"] and skip:
                pyr=smallpyr
        mcost=m["cost"].astype(numpy.float32)
        m1=m["ww"][0]
        #numy=m["ww"][0].shape[0]
        #numx=m["ww"][0].shape[1]
        for r in range(len(pyr)):
            m2=pyr[r]
            #print numy,numx
            if bbox!=None:
                nbbox=numpy.array(bbox)*f.scale[r]
            lscr,fres=crf3.match_fullN(m1,m2,N,E,mcost,show=False,feat=False,rot=False,numhyp=numhyp,aiter=aiter,restart=restart,trunc=trunc,bbox=nbbox,useFastDP=useFastDP)
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
                shift=0
                if m.has_key("big"):
                    if m["big"] and skip:
                        shift=interv-2
                det.append({"id":m["id"],"hog":r+shift,"scl":fscl[r+shift],"def":fres[idt],"scr":lscr[idt]-models[idm]["rho"]})
    det.sort(key=lambda by: -by["scr"])
    #if cfg.show:
    #    pylab.draw()
    #    pylab.show()
    print "Number of detections:",len(det)
    return [f,det]


def rundetbb(img,N,E,models,minthr=-1000,numdet=50,interv=10,aiter=3,restart=0,trunc=0,sort=True,useFastDP=False):
    "run the CRF optimization at each level of the HOG pyramid but using branch and bound algorithm"
    #note that branch and bound sometime is generating more than once the same hipothesis
    # I do not know yet why...
    #Maybe the punishment to repeat a location is not high enough
    #print "Branc and bound"
    f=pyrHOG2.pyrHOG(img,interv=interv,savedir="",hallucinate=True,cformat=True)
    ldet2=[]
    fscl=f.scale
    for idm,m in enumerate(models):
        mcost=m["cost"].astype(numpy.float32)
        m1=m["ww"][0]
        #numy=m["ww"][0].shape[0]
        #numx=m["ww"][0].shape[1]
        if minthr=="Learned":
            thr=m["thr"]+models[idm]["rho"]
        else:
            thr=minthr+models[idm]["rho"]
        pyr=f.hog
        if m.has_key("big"):
            if m["big"]:
                pyr=f.hog[interv:]
        ldet=crf3.match_bbN(m1,pyr,N,E,mcost,minthr=thr,show=False,rot=False,numhyp=numdet,aiter=aiter,restart=restart,trunc=trunc,useFastDP=useFastDP)
        for l in ldet:
            shift=0
            if m.has_key("big"):            
                if m["big"]:
                    shift=interv
            r=l["scl"]
            ldet2.append({"id":m["id"],"hog":r+shift,"scl":fscl[r+shift],"def":l["def"][0],"scr":l["scr"]-models[idm]["rho"]})            
    if sort:
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
















