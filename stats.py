import numpy
import database
import util
import pyrHOG2

def build_components(trPosImages,cfg):

    name,bb,r,a=database.extractInfo(trPosImages)
    trpos={"name":name,"bb":bb,"ratio":r,"area":a}
    import scipy.cluster.vq as vq
    numcl=cfg.numcl
    perc=cfg.perc#10
    minres=4#10
    minfy=2#3
    minfx=2#3
    #number of maximum number of HOG blocks (HOG cells /4) to use
    #maxArea=45#*(4-cfg.lev[0])#too high resolution very slow
    #maxArea=35#*(4-cfg.lev[0]) #the right trade-off
    #maxArea=25#*(4-cfg.lev[0]) #used in the test
    maxArea=cfg.maxHOG
    #maxArea=15#*(4-cfg.lev[0])
    usekmeans=False
    nh=cfg.N #number of hogs per part (for the moment everything works only with 2)

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
        meanA=numpy.mean(a[cl==l])/16.0/float(nh*nh)#(4**(cfg.lev[l]-1))#4.0
        print "Mean Area:",meanA
        sa=numpy.sort(a[cl==l])
        #minA=numpy.mean(sa[len(sa)/perc])/16.0/(0.5*4**(cfg.lev[l]-1))#4.0
        #minA=numpy.mean(sa[int(len(sa)*perc)])/4.0/(4**(cfg.lev[l]-1))#4.0
        minA=numpy.mean(sa[int(len(sa)*perc)])/16.0/float(nh*nh)#(4**(cfg.lev[l]-1))#4.0
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

    return lfy,lfx

def build_components_2(trPosImages,cfg):

    name,bb,r,a=database.extractInfo(trPosImages)
    trpos={"name":name,"bb":bb,"ratio":r,"area":a}
    import scipy.cluster.vq as vq
    numcl=cfg.numcl
    perc=cfg.perc#10
    minres=12#10
    minfy=3#3
    minfx=3#3
    #number of maximum number of HOG blocks (HOG cells /4) to use
    #maxArea=45#*(4-cfg.lev[0])#too high resolution very slow
    #maxArea=35#*(4-cfg.lev[0]) #the right trade-off
    #maxArea=25#*(4-cfg.lev[0]) #used in the test
    maxArea=cfg.maxHOG
    #maxArea=15#*(4-cfg.lev[0])
    usekmeans=False
    nh=cfg.N #number of hogs per part (for the moment everything works only with 2)
    sr=numpy.sort(r)
    spl=[]
    lfy=[];lfx=[];sfy=[];sfx=[]
    splitA=[]
    cl=numpy.zeros(r.shape)
    for l in range(numcl):
        spl.append(sr[round(l*len(r)/float(numcl))])
    spl.append(sr[-1])
    for l in range(numcl):
        cl[numpy.logical_and(r>=spl[l],r<=spl[l+1])]=l
    for l in range(numcl):
        print "Cluster same number",l,":"
        print "Samples:",len(a[cl==l])
        #meanA=numpy.mean(a[cl==l])/16.0/(0.5*4**(cfg.lev[l]-1))#4.0
        #meanA=numpy.mean(a[cl==l])/4.0/(4**(cfg.lev[l]-1))#4.0
        meanA=numpy.mean(a[cl==l])/16.0/float(nh*nh)#(4**(cfg.lev[l]-1))#4.0
        print "Mean Area:",meanA
        sa=numpy.sort(a[cl==l])
        #minA=numpy.mean(sa[len(sa)/perc])/16.0/(0.5*4**(cfg.lev[l]-1))#4.0
        #minA=num hh,_,_=pylab.hist(numpy.log(r[cl==l]),50)py.mean(sa[int(len(sa)*perc)])/4.0/(4**(cfg.lev[l]-1))#4.0
        minA=numpy.mean(sa[int(len(sa)*perc)])/16.0/float(nh*nh)#(4**(cfg.lev[l]-1))#4.0 
        print "Min Area:",minA
        #g=util.gaussian(2.0,(1,13))[0,:]
        #hh,bb=numpy.histogram(numpy.log(r[cl==l]),50)
        #hh1=numpy.convolve(hh,g,"same")
        #pbest=numpy.argmax(hh1)
        #aspt=numpy.exp((bb[pbest]+bb[min(50,pbest+1)])/2)
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
        #if minA<12:
        print"Create an additonal component for small samples"
        smallA=numpy.mean(sa[:int(len(sa)*perc/2.0)])/16.0/float(nh*nh)
        if smallA>maxArea/2:
            smallA=maxArea/2
        if aspt>1:
            fx=(max(minfx-1,numpy.sqrt(smallA/aspt)))
            fy=(fx*aspt)
        else:
            fy=(max(minfy-1,numpy.sqrt(smallA*(aspt))))
            fx=(fy/(aspt))   
        print "Fy:%.2f"%fy,"~",round(fy),"Fx:%.2f"%fx,"~",round(fx)   
        sfy.append(round(fy))
        sfx.append(round(fx))
        splitA.append(minA)
    #raw_input()    
    return lfy+sfy,lfx+sfx,splitA

def collec_posamples(trPosImagesNoTrunc):
    lfy=cfg.fy[:cfg.numcl];lfx=cfg.fx[:cfg.numcl]
    check = False
    dratios=numpy.array(lfy)/numpy.array(lfx)
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
            crop=aim[max(0,bb[0]-imy/lfy[idm]/2):min(bb[2]+imy/lfy[idm]/2,aim.shape[0]),max(0,bb[1]-imx/lfx[idm]/2):min(bb[3]+imx/lfx[idm]/2,aim.shape[1])]
            #crop=extra.getfeat(aim,abb[0]-imy/(lfy[idm]*2),bb[2]+imy/(lfy[idm]*2),bb[1]-imx/(cfg.fx[idm]*2),bb[3]+imx/(cfg.fx[idm]*2))
            imy=crop.shape[0]
            imx=crop.shape[1]
            zcim=zoom(crop,(((lfy[idm]*cfg.N+2)*8/float(imy)),((lfx[idm]*cfg.N+2)*8/float(imx)),1),order=1)
            hogp[idm].append(numpy.ascontiguousarray(pyrHOG2.hog(zcim)))
            if cfg.trunc:
                hogp[idm][-1]=numpy.concatenate((hogp[idm][-1],numpy.zeros((hogp[idm][-1].shape[0],hogp[idm][-1].shape[1],1))),2)
            #hogpcl.append(idm)
            annp[idm].append({"file":im["name"],"bbox":bb})
            if check:
                print "Aspect:",idm,"Det Size",lfy[idm]*cfg.N,lfx[idm]*cfg.N,"Shape:",zcim.shape
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
    return hogp

def collec_posamples_2(minA,trPosImagesNoTrunc,cfg):
    lfy=cfg.fy;lfx=cfg.fx
    check = False
    dratios=numpy.array(lfy[:cfg.numcl])/numpy.array(lfx[:cfg.numcl])
    hogp=[[] for x in range(cfg.numcl*2)]
    #hogpcl=[]
    annp=[[] for x in range(cfg.numcl*2)]

    #from scipy.ndimage import zoom
    from extra import myzoom as zoom
    for im in trPosImagesNoTrunc: # for each image

        #print im["name"]
        aim=util.myimread(im["name"])  
        for bb in im["bbox"]: # for each bbox (y1,x1,y2,x2)
            imy=bb[2]-bb[0]
            imx=bb[3]-bb[1]
            cropratio= imy/float(imx)
            #select the right model based on aspect ratio
            idm=numpy.argmin(abs(dratios-cropratio))
            area=imy*imx
            usesmall=False
            if area/16/cfg.N**2<minA[idm]:#use small model if there
                idm=idm+cfg.numcl
            fy=lfy[idm];fx=lfx[idm]
            crop=aim[max(0,bb[0]-imy/fy/2):min(bb[2]+imy/fy/2,aim.shape[0]),max(0,bb[1]-imx/fx/2):min(bb[3]+imx/fx/2,aim.shape[1])]
            #crop=extra.getfeat(aim,abb[0]-imy/(lfy[idm]*2),bb[2]+imy/(lfy[idm]*2),bb[1]-imx/(cfg.fx[idm]*2),bb[3]+imx/(cfg.fx[idm]*2))
            imy=crop.shape[0]
            imx=crop.shape[1]
            zcim=zoom(crop,(((fy*cfg.N+2)*8/float(imy)),((fx*cfg.N+2)*8/float(imx)),1),order=1)
            hogp[idm].append(numpy.ascontiguousarray(pyrHOG2.hog(zcim)))
            if cfg.trunc:
                hogp[idm][-1]=numpy.concatenate((hogp[idm][-1],numpy.zeros((hogp[idm][-1].shape[0],hogp[idm][-1].shape[1],1))),2)
            #hogpcl.append(idm)
            annp[idm].append({"file":im["name"],"bbox":bb})
            if check:
                print "Aspect:",idm,"Det Size",fy*cfg.N,fx*cfg.N,"Shape:",zcim.shape
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
    return hogp


