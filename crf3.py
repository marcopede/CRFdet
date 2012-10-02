import ctypes
import numpy
from numpy import ctypeslib
from ctypes import c_int,c_double,c_float
import time

#compute(int width,int height,dtype *data,int *result)
ctypes.cdll.LoadLibrary("./libcrf2.so")
lib= ctypes.CDLL("libcrf2.so")

lib.compute_graph2.argtypes=[
    #numpy.ctypeslib.ndpointer(dtype=c_int,ndim=2,flags="C_CONTIGUOUS"),# topolgy of the graph
    c_int,c_int,# num parts
    #numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),# costs for each edge
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),# costs for each edge
    c_int,# num_lab_y
    c_int,# num_lab_x
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS"),#observations
    c_int,#num hypotheses
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS"),
    numpy.ctypeslib.ndpointer(dtype=c_int,ndim=3,flags="C_CONTIGUOUS"),#labels
    c_int,#aiter
    c_int#restart
    ]
lib.compute_graph2.restype=ctypes.c_float

crfgr2=lib.compute_graph2


def match_slow(m1,m2,cost,padvalue=0,pad=0,feat=True,show=True):
    t=time.time()
    blk1=numpy.concatenate((m1[:-1:,:-1],m1[:-1,1:],m1[1:,:-1],m1[1:,1:]),2)
    blk2=numpy.concatenate((m2[:-1:,:-1],m2[:-1,1:],m2[1:,:-1],m2[1:,1:]),2)
    p1=blk1[::2,::2]
    p2=blk2[::2,::2]
    numy=p1.shape[0]
    numx=p1.shape[1]
    #numlab=blk1.shape[0]*blk1.shape[1]
    bb1=blk1.reshape((blk1.shape[0]*blk1.shape[1],-1))
    pp1=p1.reshape((p1.shape[0]*p1.shape[1],-1))
    bb2=blk2.reshape((blk2.shape[0]*blk2.shape[1],-1)).T
    #pp2=p2.reshape((p2.shape[0]*p2.shape[1],-1))

    movy=blk1.shape[0]/2
    movx=blk1.shape[1]/2
    numlab=(movy*2+1)*(movx*2+1)
    blk2pad=padvalue*numpy.ones((blk1.shape[0]+2*movy,blk1.shape[1]+2*movx,blk1.shape[2]))
    blk2pad[movy-pad:-movy+pad,movx-pad:-movx+pad]=blk2
    #data=numpy.zeros((numy,numx,(movy*2+1),(movx*2+1)),dtype=numpy.float32)
    data=numpy.zeros((numy,numx,(movy*2+1)*(movx*2+1)),dtype=numpy.float32)
    for px in range(p1.shape[1]):
        for py in range(p1.shape[0]):
            data[py,px]=-numpy.sum(p1[py,px]*blk2pad[2*py:2*py+(2*movy+1),2*px:2*px+(2*movx+1)],2).flatten()
    #print "time hog",time.time()-t
    rdata=data.reshape((data.shape[0]*data.shape[1],-1))
    rmin=rdata.min()
    rdata=rdata-rmin

#        data1=-numpy.dot(pp1,bb2)
#        t=time.time()
#        data2=-numpy.dot(pp1,bb1.T)
#        print "time hog",time.time()-t
    res=numpy.zeros((numy,numx),dtype=c_int)
    #rdata=numpy.ascontiguousarray((rdata.T-rdata.min(1).reshape((1,-1))).T)
    #sdf
    t=time.time()
    #print "before"
    scr=crfgr(numy,numx,cost,movy*2+1,movx*2+1,rdata,res)  
    #print "after"
    scr=scr-rmin*numy*numx
    res2=numpy.zeros((2,res.shape[0],res.shape[1]),dtype=c_int)
    res2[0]=(res/(movx*2+1)-movy)
    res2[1]=(res%(movx*2+1)-movx)

    if show:    
        print "time config",time.time()-t
        print scr,res
        import pylab
        pylab.figure(10)
        pylab.clf()
        pylab.ion()
        #pylab.axis("image")
        aa=pylab.axis()
        pylab.axis([aa[0],aa[1],aa[3],aa[2]])
        import util
        for px in range(res.shape[1]):
            for py in range(res.shape[0]):
                util.box(py*20+(res[py,px]/(movy*2+1)-movy)*10, px*20+(res[py,px]%(movx*2+1)-movx)*10, py*20+(res[py,px]/(movy*2+1)-movy)*10+20, px*20+(res[py,px]%(movx*2+1)-movx)*10+20, col='b', lw=2)   
                pylab.text(px*20+(res[py,px]%(movx*2+1)-movx)*10,py*20+(res[py,px]/(movy*2+1)-movy)*10,"(%d,%d)"%(py,px))
           
    if feat:
        dfeat=numpy.zeros(m1.shape,dtype=numpy.float32)
        for px in range(p1.shape[1]):
            for py in range(p1.shape[0]):
                dfeat[py*2:py*2+2,px*2:px*2+2]=blk2pad[py*2+res[py,px]/(movx*2+1),px*2+res[py,px]%(movx*2+1)].reshape(2,2,31)
        edge=numpy.zeros(res2.shape,dtype=numpy.float32)
        #edge[0,:-1,:]=(res2[0,:-1]-res2[0,1:])
        #edge[1,:,:-1]=(res2[1,:,:-1]-res2[1,:,1:])
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])+abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[1,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])+abs(res2[1,:,:-1]-res2[1,:,1:])
        #edge[0,:-1,:]=abs(dy)/(movx*2+1)+abs(dy)%(movx*2+1)
        #edge[1,:,:-1]=abs(dx)/(movx*2+1)+abs(dx)%(movx*2+1)
        return scr,res2,dfeat,-edge

    return scr,res

from ctypes import c_int,c_double,c_float
import ctypes
ctypes.cdll.LoadLibrary("./libexcorr.so")
ff= ctypes.CDLL("libexcorr.so")
ff.scaneigh.argtypes=[numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),c_int,c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer
(dtype=c_int,flags="C_CONTIGUOUS"),c_int,c_int,c_int,c_int]

#inline ftype refineighfull(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,ftype dy,ftype dx,int posy,int posx,int rady,int radx,ftype *scr,int *rdy,int *rdx,ftype *prec,int pady,int padx,int occl)
ff.refineighfull.argtypes=[
numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,
numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),c_int,c_int,c_int,
c_float,c_float,c_int,c_int,c_int,c_int,
numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS"),
numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS"),
numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS"),
ctypes.POINTER(c_float),c_int,c_int,c_int]
ff.refineighfull.restype=c_float

def match_old(m1,m2,cost,movy=None,movx=None,padvalue=0,pad=0,feat=True,show=True,rotate=False):
    t=time.time()
    #blk1=numpy.concatenate((m1[:-1:,:-1],m1[:-1,1:],m1[1:,:-1],m1[1:,1:]),2)
    #blk2=numpy.concatenate((m2[:-1:,:-1],m2[:-1,1:],m2[1:,:-1],m2[1:,1:]),2)
    #p1=blk1[::2,::2]
    numy=m1.shape[0]/2#p1.shape[0]
    numx=m1.shape[1]/2#p1.shape[1]
    #bb2=blk2.reshape((blk2.shape[0]*blk2.shape[1],-1)).T

    if movy==None:
        movy=((m1.shape[0]-2*pad)-1)/2
    if movx==None:
        movx=((m1.shape[1]-2*pad)-1)/2
    numlab=(movy*2+1)*(movx*2+1)
    #blk2pad=padvalue*numpy.ones((blk1.shape[0]+2*movy,blk1.shape[1]+2*movx,blk1.shape[2]))
    #blk2pad[movy-pad:-movy+pad,movx-pad:-movx+pad]=blk2
    data=numpy.zeros((numy,numx,(movy*2+1)*(movx*2+1)),dtype=numpy.float32)
    #print "time preparation",time.time()-t
    t=time.time()
#    t1=time.time()
#    for px in range(p1.shape[1]):
#        for py in range(p1.shape[0]):
#            data[py,px]=-numpy.sum(p1[py,px]*blk2pad[2*py:2*py+(2*movy+1),2*px:2*px+(2*movx+1)],2).flatten()
#    print "Time mode1",time.time()-t1

    #data1=numpy.zeros((numy,numx,(movy*2+1)*(movx*2+1)),dtype=numpy.float32)
    mmax=numpy.zeros(2,dtype=c_int)
    #t1=time.time()
    #m1=numpy.ascontiguousarray(m1)
    m2=numpy.ascontiguousarray(m2)
    for px in range(numx):
        for py in range(numy):
            ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,data[py,px],
                mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)
    #print "Time mode1",time.time()-t1
    
    #print "time hog",time.time()-t
    rdata=data.reshape((data.shape[0]*data.shape[1],-1))
    rmin=rdata.min()
    rdata=rdata-rmin

    res=numpy.zeros((numy,numx),dtype=c_int)
    #print "time matching",time.time()-t
    t=time.time()
    #print "before"
    scr=crfgr(numy,numx,cost,movy*2+1,movx*2+1,rdata,res)  
    #print "after"
    scr=scr-rmin*numy*numx
    res2=numpy.zeros((2,res.shape[0],res.shape[1]),dtype=c_int)
    res2[0]=(res/(movx*2+1)-movy)
    res2[1]=(res%(movx*2+1)-movx)
    #print "time config",time.time()-t

    if show:    
        print scr,res
        import pylab
        pylab.figure(10)
        pylab.clf()
        pylab.ion()
        #pylab.axis("image")
        aa=pylab.axis()
        pylab.axis([aa[0],aa[1],aa[3],aa[2]])
        import util
        for px in range(res.shape[1]):
            for py in range(res.shape[0]):
                util.box(py*20+(res[py,px]/(movy*2+1)-movy)*10, px*20+(res[py,px]%(movx*2+1)-movx)*10, py*20+(res[py,px]/(movy*2+1)-movy)*10+20, px*20+(res[py,px]%(movx*2+1)-movx)*10+20, col='b', lw=2)   
                pylab.text(px*20+(res[py,px]%(movx*2+1)-movx)*10,py*20+(res[py,px]/(movy*2+1)-movy)*10,"(%d,%d)"%(py,px))
           
    if feat:
        t=time.time()
        dfeat=numpy.zeros(m1.shape,dtype=numpy.float32)
        m2pad=numpy.zeros((m2.shape[0]+2*movy-2*pad,m2.shape[1]+2*movx-2*pad,m2.shape[2]),dtype=numpy.float32)
        m2pad[movy-pad:-movy+pad,movx-pad:-movx+pad]=m2
        for px in range(numx):
            for py in range(numy):
                #dfeat[py*2:py*2+2,px*2:px*2+2]=blk2pad[py*2+res[py,px]/(movx*2+1),px*2+res[py,px]%(movx*2+1)].reshape(2,2,31)
                cpy=py*2+res2[0,py,px]+movy
                cpx=px*2+res2[1,py,px]+movx    
                dfeat[py*2:py*2+2,px*2:px*2+2]=m2pad[cpy:cpy+2,cpx:cpx+2]
        edge=numpy.zeros(res2.shape,dtype=numpy.float32)
        #edge[0,:-1,:]=(res2[0,:-1]-res2[0,1:])
        #edge[1,:,:-1]=(res2[1,:,:-1]-res2[1,:,1:])
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])+abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[1,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])+abs(res2[1,:,:-1]-res2[1,:,1:])
        #edge[0,:-1,:]=abs(dy)/(movx*2+1)+abs(dy)%(movx*2+1)
        #edge[1,:,:-1]=abs(dx)/(movx*2+1)+abs(dx)%(movx*2+1)
        #print "time feat",time.time()-t
        return scr,res2,dfeat,-edge

    return scr,res2

def rotate(hog,shift=1):
    """
    rotate each hog cell of a certain shift
    """
    if shift==0:
        return hog
    hbin=9
    rhog=numpy.zeros(hog.shape,dtype=numpy.float32)
    rhog[:,:,:18]=hog[:,:,numpy.mod(numpy.arange(shift,hbin*2+shift),hbin*2)]
    rhog[:,:,18:27]=hog[:,:,numpy.mod(numpy.arange(shift,hbin+shift),hbin)+18]
    rhog[:,:,27:]=hog[:,:,27:]
    return rhog

def box(p1y,p1x,p2y,p2x,col='b',lw=1,rot=0):
    """
        plot a bbox with the given coordinates
    """
    import pylab

    rrot=rot*numpy.pi/180.0
    cy=(p2y+p1y)/2.0
    cx=(p2x+p1x)/2.0
    #print cy,cx
    from matplotlib.transforms import Affine2D
    r=Affine2D().rotate_around(cx,cy,rrot)
    p1=numpy.dot(r,[p1x,p1y,1])
    p2=numpy.dot(r,[p1x,p2y,1])
    p3=numpy.dot(r,[p2x,p2y,1])
    p4=numpy.dot(r,[p2x,p1y,1])
    #print r,p1,p2,p3,p4
    #np1y=p1y+(p1y-cy)*numpy.sin(rrot)+(p1x-cx)*numpy.cos(rrot)
    #np1x=p1x+(p1y-cy)*numpy.cos(rrot)+(p1x-cx)*numpy.sin(rrot)
    #np2y=p2y+(p2y-cy)*numpy.sin(rrot)+(p2x-cx)*numpy.cos(rrot)
    #np2x=p2x+(p2y-cy)*numpy.cos(rrot)+(p1x-cx)*numpy.sin(rrot)
    pylab.plot([p1[0],p2[0],p3[0],p4[0],p1[0]],[p1[1],p2[1],p3[1],p4[1],p1[1]],col,lw=lw)
    #pylab.fill([p1x,p1x,p2x,p2x,p1x],[p1y,p2y,p2y,p1y,p1y],col,lw=lw)


def match(m1,m2,cost,movy=None,movx=None,padvalue=0,pad=0,feat=True,show=True,rot=False):
    t=time.time()
    numy=m1.shape[0]/2#p1.shape[0]
    numx=m1.shape[1]/2#p1.shape[1]
    if movy==None:
        movy=((m1.shape[0]-2*pad)-1)/2
    if movx==None:
        movx=((m1.shape[1]-2*pad)-1)/2
    numlab=(movy*2+1)*(movx*2+1)
    data=numpy.zeros((numy,numx,(movy*2+1)*(movx*2+1)),dtype=numpy.float32)
    auxdata=numpy.zeros((3,numy,numx,(movy*2+1)*(movx*2+1)),dtype=numpy.float32)
    #print "time preparation",time.time()-t
    t=time.time()
    mmax=numpy.zeros(2,dtype=c_int)
    #original model
    m2=numpy.ascontiguousarray(m2)
    for px in range(numx):
        for py in range(numy):
            ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[1,py,px],
                mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)
    if rot:
        #rotate +1
        m2=numpy.ascontiguousarray(rotate(m2,shift=1))
        for px in range(numx):
            for py in range(numy):
                ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                    2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[2,py,px],
                    mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)
        #rotate -1
        m2=numpy.ascontiguousarray(rotate(m2,shift=-2))
        for px in range(numx):
            for py in range(numy):
                ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                    2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[0,py,px],
                    mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)

        #print "time hog",time.time()-t
        data=numpy.min(auxdata,0)
        mrot=numpy.argmin(auxdata,0)
        #print rot
        rdata=data.reshape((data.shape[0]*data.shape[1],-1))
        #rrot=rot.reshape((rot.shape[0]*rot.shape[1],-1))
    else:
        data=auxdata[1]
        rdata=data.reshape((data.shape[0]*data.shape[1],-1))
    rmin=rdata.min()
    rdata=rdata-rmin

    res=numpy.zeros((numy,numx),dtype=c_int)
    #print "time matching",time.time()-t
    t=time.time()
    #print "before"
    scr=crfgr(numy,numx,cost,movy*2+1,movx*2+1,rdata,res)  
    #print "after"
    scr=scr-rmin*numy*numx
    res2=numpy.zeros((2,res.shape[0],res.shape[1]),dtype=c_int)
    res2[0]=(res/(movx*2+1)-movy)
    res2[1]=(res%(movx*2+1)-movx)
    print "time config",time.time()-t
    if rot:
        frot=numpy.zeros((mrot.shape[0],mrot.shape[1]),dtype=numpy.int32)
        for py in range(res.shape[0]):
            for px in range(res.shape[1]):
                frot[py,px]=mrot[py,px,res[py,px]]-1
        return scr,res2,frot

    return scr,res2

def match_full(m1,m2,cost,movy=None,movx=None,padvalue=0,remove=[],pad=0,feat=True,show=True,rot=False):
    t=time.time()
    numy=m1.shape[0]/2#p1.shape[0]
    numx=m1.shape[1]/2#p1.shape[1]
    print numy,numx
    if movy==None:
        movy=(m2.shape[0]+m1.shape[0])
    if movx==None:
        movx=(m2.shape[1]+m1.shape[1])
    numlab=movy*movx
    data=numpy.zeros((numy,numx,numlab),dtype=numpy.float32)
    auxdata=numpy.zeros((3,numy,numx,numlab),dtype=numpy.float32)
    #print "time preparation",time.time()-t
    t=time.time()
    mmax=numpy.zeros(2,dtype=c_int)
    #original model
    m2=numpy.ascontiguousarray(m2)
    scn=numpy.mgrid[:movy,:movx].astype(numpy.int32)
    scn[0]=scn[0]-m1.shape[0]
    scn[1]=scn[1]-m1.shape[1]
    tmp=scn.copy()
    for px in range(numx):
        for py in range(numy):
            #ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]), 2,2,m1.shape[2],0,0,py*2-m1.shape[0],px*2-m1.shape[1],movy/2,movx/2,auxdata[1,py,px],          mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)
            ff.scaneigh(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]), 2,2,m1.shape[2],scn[0]+2*py,scn[1]+2*px,auxdata[1,py,px],tmp[0],tmp[1],0,0,scn[0].size,0)
            #print "Done ",py,px 
    print "time Match",time.time()-t
    if rot:
        #rotate +1
        m2=numpy.ascontiguousarray(rotate(m2,shift=1))
        for px in range(numx):
            for py in range(numy):
                ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                    2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[2,py,px],
                    mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)
        #rotate -1
        m2=numpy.ascontiguousarray(rotate(m2,shift=-2))
        for px in range(numx):
            for py in range(numy):
                ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                    2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[0,py,px],
                    mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)

        #print "time hog",time.time()-t
        data=numpy.min(auxdata,0)
        mrot=numpy.argmin(auxdata,0)
        #print rot
        rdata=data.reshape((data.shape[0]*data.shape[1],-1))
        #rrot=rot.reshape((rot.shape[0]*rot.shape[1],-1))
    else:
        data=-auxdata[1]
        if remove!=[]: # remove all elements already computed
            for rm in remove:
                for py in range(rm.shape[1]):
                    for px in range(rm.shape[2]):
                        rcy=rm[0,py,px]+m1.shape[0]
                        rcx=rm[1,py,px]+m1.shape[1]
                        data.reshape((data.shape[0],data.shape[1],movy,movx))[py,px,rcy,rcx]=1
                        #data.reshape((data.shape[0],data.shape[1],movy,movx))[py,px,rcy-1:rcy+2,rcx-1:rcx+2]=1
        rdata=data.reshape((data.shape[0]*data.shape[1],-1))
    rmin=rdata.min()
    rdata=rdata-rmin

    res=numpy.zeros((numy,numx),dtype=c_int)
    #print "time matching",time.time()-t
    t=time.time()
    #print "before",rdata.sum()
    #order=numpy.arange(movy*movx,dtype=numpy.int32)
    order=numpy.argsort(-numpy.sum(rdata,0)).astype(numpy.int32)
    if 1:
        import pylab
        pylab.figure()
        detim=-numpy.sum(rdata,0).reshape((movy,movx))
        pylab.imshow(detim,interpolation="nearest")
        print "Rigid Best Value",detim.max()-rmin*numy*numx
        print "Max Location",numpy.where(detim==detim.max())
        #print (-rdata.reshape(data.shape[0],data.shape[1],-1)).max(2)
        print "Unconctrained Best Value",numpy.sum((-rdata.reshape(data.shape[0],data.shape[1],-1)).max(2))-rmin*numy*numx

    scr=crfgr(numy,numx,cost,movy,movx,rdata,order,res)  
    #print "after",rdata.sum()
    scr=scr-rmin*numy*numx
    res2=numpy.zeros((2,res.shape[0],res.shape[1]),dtype=c_int)
    res2[0]=(res/(movx))-m1.shape[0]
    res2[1]=(res%(movx))-m1.shape[1]
    print "time config",time.time()-t
    if rot:
        frot=numpy.zeros((mrot.shape[0],mrot.shape[1]),dtype=numpy.int32)
        for py in range(res.shape[0]):
            for px in range(res.shape[1]):
                frot[py,px]=mrot[py,px,res[py,px]]-1
        return scr,res2,frot
    return scr,res2


def match_full2(m1,m2,cost,movy=None,movx=None,padvalue=0,remove=[],pad=0,feat=True,show=True,rot=False,    numhyp=10,output=False,aiter=3,restart=0):
    t=time.time()
    numy=m1.shape[0]/2#p1.shape[0]
    numx=m1.shape[1]/2#p1.shape[1]
    #print numy,numx
    if movy==None:
        movy=(m2.shape[0]+m1.shape[0])
    if movx==None:
        movx=(m2.shape[1]+m1.shape[1])
    numlab=movy*movx
    data=numpy.zeros((numy,numx,numlab),dtype=c_float)
    auxdata=numpy.zeros((3,numy,numx,numlab),dtype=c_float)
    #print "time preparation",time.time()-t
    t=time.time()
    mmax=numpy.zeros(2,dtype=c_int)
    #original model
    m2=numpy.ascontiguousarray(m2)
    scn=numpy.mgrid[:movy,:movx].astype(numpy.int32)
    scn[0]=scn[0]-m1.shape[0]
    scn[1]=scn[1]-m1.shape[1]
    tmp=scn.copy()
    for px in range(numx):
        for py in range(numy):
            #ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]), 2,2,m1.shape[2],0,0,py*2-m1.shape[0],px*2-m1.shape[1],movy/2,movx/2,auxdata[1,py,px],          mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)
            ff.scaneigh(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]), 2,2,m1.shape[2],scn[0]+2*py,scn[1]+2*px,auxdata[1,py,px],tmp[0],tmp[1],0,0,scn[0].size,0)
            #print "Done ",py,px 
    if output:
        print "time Match",time.time()-t
    if rot:
        #rotate +1
        m2=numpy.ascontiguousarray(rotate(m2,shift=1))
        for px in range(numx):
            for py in range(numy):
                ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                    2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[2,py,px],
                    mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)
        #rotate -1
        m2=numpy.ascontiguousarray(rotate(m2,shift=-2))
        for px in range(numx):
            for py in range(numy):
                ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                    2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[0,py,px],
                    mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)

        #print "time hog",time.time()-t
        data=numpy.min(auxdata,0)
        mrot=numpy.argmin(auxdata,0)
        #print rot
        rdata=data.reshape((data.shape[0]*data.shape[1],-1))
        #rrot=rot.reshape((rot.shape[0]*rot.shape[1],-1))
    else:
        data=-auxdata[1]
        if remove!=[]: # remove all elements already computed
            for rm in remove:
                for py in range(rm.shape[1]):
                    for px in range(rm.shape[2]):
                        rcy=rm[0,py,px]+m1.shape[0]
                        rcx=rm[1,py,px]+m1.shape[1]
                        data.reshape((data.shape[0],data.shape[1],movy,movx))[py,px,rcy,rcx]=1
                        #data.reshape((data.shape[0],data.shape[1],movy,movx))[py,px,rcy-1:rcy+2,rcx-1:rcx+2]=1
        rdata=data.reshape((data.shape[0]*data.shape[1],-1))
    rmin=rdata.min()
    rdata=rdata-rmin

    res=numpy.zeros((numhyp,numy,numx),dtype=c_int)
    #print "time matching",time.time()-t
    #print "before",rdata.sum()
    #order=numpy.arange(movy*movx,dtype=numpy.int32)
    #order=numpy.argsort(-numpy.sum(rdata,0)).astype(numpy.int32)
    if 0:
        import pylab
        pylab.figure()
        detim=-numpy.sum(rdata,0).reshape((movy,movx))
        pylab.imshow(detim,interpolation="nearest")
        print "Rigid Best Value",detim.max()-rmin*numy*numx
        print "Max Location",numpy.where(detim==detim.max())
        #print (-rdata.reshape(data.shape[0],data.shape[1],-1)).max(2)
        print "Unconstrained Best Value",numpy.sum((-rdata.reshape(data.shape[0],data.shape[1],-1)).max(2))-rmin*numy*numx

    #t=time.time()
    t=time.time()
    lscr=numpy.zeros(numhyp,dtype=numpy.float32)
    scr=crfgr2(numy,numx,cost,movy,movx,rdata,numhyp,lscr,res,aiter,restart)  
    #print "after",rdata.sum()
    scr=scr-rmin*numy*numx
    lscr=lscr-rmin*numy*numx
    res2=numpy.zeros((numhyp,2,res.shape[1],res.shape[2]),dtype=c_int)
    res2[:,0]=(res/(movx))-m1.shape[0]
    res2[:,1]=(res%(movx))-m1.shape[1]
    if output:
        print "time config",time.time()-t
    if rot:
        frot=numpy.zeros((mrot.shape[0],mrot.shape[1]),dtype=numpy.int32)
        for py in range(res.shape[1]):
            for px in range(res.shape[2]):
                frot[py,px]=mrot[py,px,res[py,px]]-1
        return scr,res2,frot
    return lscr,res2

def match_fullN(m1,m2,N,cost,remove=[],pad=0,feat=True,show=True,rot=False,    numhyp=10,output=False,aiter=3,restart=0):
    #m1 is expected to be divisible by N
    t=time.time()
    assert(m1.shape[0]%N==0)
    assert(m1.shape[1]%N==0)
    numy=m1.shape[0]/N#p1.shape[0]
    numx=m1.shape[1]/N#p1.shape[1]
    #print numy,numx
    movy=(m2.shape[0]+m1.shape[0])
    movx=(m2.shape[1]+m1.shape[1])
    numlab=movy*movx
    data=numpy.zeros((numy,numx,numlab),dtype=c_float)
    auxdata=numpy.zeros((3,numy,numx,numlab),dtype=c_float)
    #print "time preparation",time.time()-t
    t=time.time()
    mmax=numpy.zeros(2,dtype=c_int)
    #original model
    m2=numpy.ascontiguousarray(m2)
    scn=numpy.mgrid[:movy,:movx].astype(numpy.int32)
    scn[0]=scn[0]-m1.shape[0]
    scn[1]=scn[1]-m1.shape[1]
    tmp=scn.copy()
    for px in range(numx):
        for py in range(numy):
            ff.scaneigh(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[N*py:N*(py+1),N*px:N*(px+1)]),N,N,m1.shape[2],scn[0]+N*py,scn[1]+N*px,auxdata[1,py,px],tmp[0],tmp[1],0,0,scn[0].size,0)
            #print "Done ",py,px 
    if output:
        print "time Match",time.time()-t
    if rot:
        #rotate +1
        m2=numpy.ascontiguousarray(rotate(m2,shift=1))
        for px in range(numx):
            for py in range(numy):
                ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                    2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[2,py,px],
                    mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)
        #rotate -1
        m2=numpy.ascontiguousarray(rotate(m2,shift=-2))
        for px in range(numx):
            for py in range(numy):
                ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                    2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[0,py,px],
                    mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)

        #print "time hog",time.time()-t
        data=numpy.min(auxdata,0)
        mrot=numpy.argmin(auxdata,0)
        #print rot
        rdata=data.reshape((data.shape[0]*data.shape[1],-1))
        #rrot=rot.reshape((rot.shape[0]*rot.shape[1],-1))
    else:
        data=-auxdata[1]
        rdata=data.reshape((data.shape[0]*data.shape[1],-1))
    rmin=rdata.min()
    rdata=rdata-rmin

    res=numpy.zeros((numhyp,numy,numx),dtype=c_int)
    #print "time matching",time.time()-t
    #print "before",rdata.sum()
    #order=numpy.arange(movy*movx,dtype=numpy.int32)
    #order=numpy.argsort(-numpy.sum(rdata,0)).astype(numpy.int32)
    if 0:
        import pylab
        pylab.figure()
        detim=-numpy.sum(rdata,0).reshape((movy,movx))
        pylab.imshow(detim,interpolation="nearest")
        print "Rigid Best Value",detim.max()-rmin*numy*numx
        print "Max Location",numpy.where(detim==detim.max())
        #print (-rdata.reshape(data.shape[0],data.shape[1],-1)).max(2)
        print "Unconstrained Best Value",numpy.sum((-rdata.reshape(data.shape[0],data.shape[1],-1)).max(2))-rmin*numy*numx

    #t=time.time()
    t=time.time()
    lscr=numpy.zeros(numhyp,dtype=numpy.float32)
    scr=crfgr2(numy,numx,cost,movy,movx,rdata,numhyp,lscr,res,aiter,restart)  
    #print "after",rdata.sum()
    scr=scr-rmin*numy*numx
    lscr=lscr-rmin*numy*numx
    res2=numpy.zeros((numhyp,2,res.shape[1],res.shape[2]),dtype=c_int)
    res2[:,0]=(res/(movx))-m1.shape[0]
    res2[:,1]=(res%(movx))-m1.shape[1]
    if output:
        print "time config",time.time()-t
    if rot:
        frot=numpy.zeros((mrot.shape[0],mrot.shape[1]),dtype=numpy.int32)
        for py in range(res.shape[1]):
            for px in range(res.shape[2]):
                frot[py,px]=mrot[py,px,res[py,px]]-1
        return scr,res2,frot
    return lscr,res2


def match_bb(m1,pm2,cost,show=True,rot=False,numhyp=10,aiter=3,restart=0):
    t=time.time()
    numy=m1.shape[0]/2#p1.shape[0]
    numx=m1.shape[1]/2#p1.shape[1]
    #print numy,numx
    data=[];minb=[];maxb=[]
    for idm2,m2 in enumerate(pm2):
        movy=(m2.shape[0]+m1.shape[0])
        movx=(m2.shape[1]+m1.shape[1])
        numlab=movy*movx
        data.append(numpy.zeros((numy,numx,numlab),dtype=c_float))
        auxdata=numpy.zeros((3,numy,numx,numlab),dtype=c_float)
        #print "time preparation",time.time()-t
        t=time.time()
        mmax=numpy.zeros(2,dtype=c_int)
        #original model
        m2=numpy.ascontiguousarray(m2)
        scn=numpy.mgrid[:movy,:movx].astype(numpy.int32)
        scn[0]=scn[0]-m1.shape[0]
        scn[1]=scn[1]-m1.shape[1]
        tmp=scn.copy()
        for px in range(numx):
            for py in range(numy):
                ff.scaneigh(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]), 2,2,m1.shape[2],scn[0]+2*py,scn[1]+2*px,auxdata[1,py,px],tmp[0],tmp[1],0,0,scn[0].size,0)
                #print "Done ",py,px 
        #if output:
        #    print "time Match",time.time()-t
        if 0:#rot:
            #rotate +1
            m2=numpy.ascontiguousarray(rotate(m2,shift=1))
            for px in range(numx):
                for py in range(numy):
                    ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                        2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[2,py,px],
                        mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)
            #rotate -1
            m2=numpy.ascontiguousarray(rotate(m2,shift=-2))
            for px in range(numx):
                for py in range(numy):
                    ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                        2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[0,py,px],
                        mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)

            #print "time hog",time.time()-t
            data=numpy.min(auxdata,0)
            mrot=numpy.argmin(auxdata,0)
            #print rot
            #rdata=data.reshape((data.shape[0]*data.shape[1],-1))
            #rrot=rot.reshape((rot.shape[0]*rot.shape[1],-1))
        else:
            data[-1]=auxdata[1]
            rrdata=data[-1].reshape((data[-1].shape[0]*data[-1].shape[1],-1))
        #estimate max and min
        minb.append(numpy.sum(rrdata,0).max())
        maxb.append(numpy.sum(rrdata.max(1)))
        ##print "Lev",idm2,"Min:",minb[-1],"Max:",maxb[-1]
        #data[-1]=-data[-1]
        #raw_input()
    maxb=numpy.array(maxb)
    minb=numpy.array(minb)
    lres=numpy.zeros(len(maxb),dtype=object)
    lscr=numpy.zeros(len(maxb))
    ldet=[]
    for h in range(numhyp):
        stop=False
        while not(stop):
            l=maxb.argmax()#the max bound
            m2=-data[l]
            #rm2=m2.reshape((data[-1].shape[0]*data[-1].shape[1],-1))
            movy=(pm2[l].shape[0]+m1.shape[0])
            movx=(pm2[l].shape[1]+m1.shape[1])
            auxmin=m2.min()
            rdata=m2-auxmin
            auxscr=numpy.zeros(1,dtype=numpy.float32)
            res=numpy.zeros((1,numy,numx),dtype=c_int)
            if 0:
                import pylab
                pylab.figure(200)
                pylab.clf()
                pylab.imshow(-rdata.reshape((rdata.shape[0]*rdata.shape[1],-1)).sum(0).reshape(movy,movx)-auxmin*numy*numx,vmin=0,vmax=3.0,interpolation="nearest")
                pylab.draw()
                pylab.show()
            scr=crfgr2(numy,numx,cost,movy,movx,rdata.reshape((rdata.shape[0]*rdata.shape[1],-1)),1,auxscr,res,aiter,restart)  
            assert(scr==auxscr[0])
            #print "Before",scr
            scr=scr-auxmin*numy*numx
            #update bounds and save detection
            ##print "Lev",l,"Old Maxb",maxb[l],"New Maxb",scr
            #assert(scr+0.00001>=minb[l])# not always true because alpha expansion doen not give the global optimum
            maxb[l]=scr
            res2=numpy.zeros((1,2,res.shape[1],res.shape[2]),dtype=c_int)
            res2[:,0]=(res/(movx))-m1.shape[0]
            res2[:,1]=(res%(movx))-m1.shape[1]
            lres[l]=res2
            lscr[l]=scr
            #assert(scr>=minb[l])
            if lscr.max()+0.00001>=maxb.max():
                stop=True
                lmax=lscr.argmax()
                ##print "Found maxima Lev",lmax,"Scr",lscr[lmax]
                #raw_input()
                ldet.append({"scl":lmax,"scr":lscr[lmax],"def":lres[lmax].copy()})
                #update data
                res2=lres[lmax]
                movy=(pm2[lmax].shape[0]+m1.shape[0])
                movx=(pm2[lmax].shape[1]+m1.shape[1])            
                for p in range(res2.shape[2]):
                    for px in range(res2.shape[3]):
                        rcy=res2[0,0,py,px]+m1.shape[0]
                        rcx=res2[0,1,py,px]+m1.shape[1]
                        #data.reshape((data.shape[0],data.shape[1],movy,movx))[py,px,rcy,rcx]=1
                        data[lmax].reshape((numy,numx,movy,movx))[py,px,rcy-1:rcy+2,rcx-1:rcx+2]=-1
                #update bounds
                minb[lmax]=numpy.max(numpy.sum(data[lmax].reshape((data[lmax].shape[0]*data[lmax].shape[1],-1)),0))
                #maxb[lmax]=numpy.sum(numpy.max(data[l],2))
   
    return ldet

def match_bbN(m1,pm2,N,cost,show=True,rot=False,numhyp=10,aiter=3,restart=0):
    t=time.time()
    assert(m1.shape[0]%N==0)
    assert(m1.shape[1]%N==0)
    numy=m1.shape[0]/N#p1.shape[0]
    numx=m1.shape[1]/N#p1.shape[1]
    #print numy,numx
    data=[];minb=[];maxb=[]
    for idm2,m2 in enumerate(pm2):
        movy=(m2.shape[0]+m1.shape[0])
        movx=(m2.shape[1]+m1.shape[1])
        numlab=movy*movx
        data.append(numpy.zeros((numy,numx,numlab),dtype=c_float))
        auxdata=numpy.zeros((3,numy,numx,numlab),dtype=c_float)
        #print "time preparation",time.time()-t
        t=time.time()
        mmax=numpy.zeros(2,dtype=c_int)
        #original model
        m2=numpy.ascontiguousarray(m2)
        scn=numpy.mgrid[:movy,:movx].astype(numpy.int32)
        scn[0]=scn[0]-m1.shape[0]
        scn[1]=scn[1]-m1.shape[1]
        tmp=scn.copy()
        for px in range(numx):
            for py in range(numy):
                ff.scaneigh(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[N*py:N*(py+1),N*px:N*(px+1)]),N,N,m1.shape[2],scn[0]+N*py,scn[1]+N*px,auxdata[1,py,px],tmp[0],tmp[1],0,0,scn[0].size,0)
                #print "Done ",py,px 
        #if output:
        #    print "time Match",time.time()-t
        if 0:#rot:
            #rotate +1
            m2=numpy.ascontiguousarray(rotate(m2,shift=1))
            for px in range(numx):
                for py in range(numy):
                    ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                        2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[2,py,px],
                        mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)
            #rotate -1
            m2=numpy.ascontiguousarray(rotate(m2,shift=-2))
            for px in range(numx):
                for py in range(numy):
                    ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                        2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[0,py,px],
                        mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)

            #print "time hog",time.time()-t
            data=numpy.min(auxdata,0)
            mrot=numpy.argmin(auxdata,0)
            #print rot
            #rdata=data.reshape((data.shape[0]*data.shape[1],-1))
            #rrot=rot.reshape((rot.shape[0]*rot.shape[1],-1))
        else:
            data[-1]=auxdata[1]
            rrdata=data[-1].reshape((data[-1].shape[0]*data[-1].shape[1],-1))
        #estimate max and min
        minb.append(numpy.sum(rrdata,0).max())
        maxb.append(numpy.sum(rrdata.max(1)))
        ##print "Lev",idm2,"Min:",minb[-1],"Max:",maxb[-1]
        #data[-1]=-data[-1]
        #raw_input()
    maxb=numpy.array(maxb)
    minb=numpy.array(minb)
    lres=numpy.zeros(len(maxb),dtype=object)
    lscr=numpy.zeros(len(maxb))
    ldet=[]
    for h in range(numhyp):
        stop=False
        while not(stop):
            l=maxb.argmax()#the max bound
            m2=-data[l]
            #rm2=m2.reshape((data[-1].shape[0]*data[-1].shape[1],-1))
            movy=(pm2[l].shape[0]+m1.shape[0])
            movx=(pm2[l].shape[1]+m1.shape[1])
            auxmin=m2.min()
            rdata=m2-auxmin
            auxscr=numpy.zeros(1,dtype=numpy.float32)
            res=numpy.zeros((1,numy,numx),dtype=c_int)
            if 0:
                import pylab
                pylab.figure(200)
                pylab.clf()
                pylab.imshow(-rdata.reshape((rdata.shape[0]*rdata.shape[1],-1)).sum(0).reshape(movy,movx)-auxmin*numy*numx,vmin=0,vmax=3.0,interpolation="nearest")
                pylab.draw()
                pylab.show()
            scr=crfgr2(numy,numx,cost,movy,movx,rdata.reshape((rdata.shape[0]*rdata.shape[1],-1)),1,auxscr,res,aiter,restart)  
            assert(scr==auxscr[0])
            #print "Before",scr
            scr=scr-auxmin*numy*numx
            #update bounds and save detection
            ##print "Lev",l,"Old Maxb",maxb[l],"New Maxb",scr
            #assert(scr+0.00001>=minb[l])# not always true because alpha expansion doen not give the global optimum
            maxb[l]=scr
            res2=numpy.zeros((1,2,res.shape[1],res.shape[2]),dtype=c_int)
            res2[:,0]=(res/(movx))-m1.shape[0]
            res2[:,1]=(res%(movx))-m1.shape[1]
            lres[l]=res2
            lscr[l]=scr
            #assert(scr>=minb[l])
            if lscr.max()+0.00001>=maxb.max():
                stop=True
                lmax=lscr.argmax()
                ##print "Found maxima Lev",lmax,"Scr",lscr[lmax]
                #raw_input()
                ldet.append({"scl":lmax,"scr":lscr[lmax],"def":lres[lmax].copy()})
                #update data
                res2=lres[lmax]
                movy=(pm2[lmax].shape[0]+m1.shape[0])
                movx=(pm2[lmax].shape[1]+m1.shape[1])            
                for p in range(res2.shape[2]):
                    for px in range(res2.shape[3]):
                        rcy=res2[0,0,py,px]+m1.shape[0]
                        rcx=res2[0,1,py,px]+m1.shape[1]
                        #data.reshape((data.shape[0],data.shape[1],movy,movx))[py,px,rcy,rcx]=1
                        data[lmax].reshape((numy,numx,movy,movx))[py,px,rcy-1:rcy+2,rcx-1:rcx+2]=-1
                #update bounds
                minb[lmax]=numpy.max(numpy.sum(data[lmax].reshape((data[lmax].shape[0]*data[lmax].shape[1],-1)),0))
                #maxb[lmax]=numpy.sum(numpy.max(data[l],2))
   
    return ldet


def getfeat(m1,pad,res2,movy=None,movx=None,mode="Best",rot=None):
    if movy==None:
        movy=((m1.shape[0]-2*pad)-1)/2
    if movx==None:
        movx=((m1.shape[1]-2*pad)-1)/2
    dfeat=numpy.zeros((m1.shape[0]-2*pad,m1.shape[1]-2*pad,m1.shape[2]),dtype=numpy.float32)
    m1pad=numpy.zeros((m1.shape[0]+2*movy-2*pad,m1.shape[1]+2*movx-2*pad,m1.shape[2]),dtype=numpy.float32)
    m1pad[movy-pad:-movy+pad,movx-pad:-movx+pad]=m1
    for px in range(res2.shape[2]):
        for py in range(res2.shape[1]):
            #dfeat[py*2:py*2+2,px*2:px*2+2]=blk2pad[py*2+res[py,px]/(movx*2+1),px*2+res[py,px]%(movx*2+1)].reshape(2,2,31)
            cpy=py*2+res2[0,py,px]+movy
            cpx=px*2+res2[1,py,px]+movx    
            if rot==None:
                dfeat[py*2:py*2+2,px*2:px*2+2]=m1pad[cpy:cpy+2,cpx:cpx+2]
            else:
                dfeat[py*2:py*2+2,px*2:px*2+2]=rotate(m1pad[cpy:cpy+2,cpx:cpx+2],rot[py,px])
    edge=numpy.zeros((res2.shape[0]*2,res2.shape[1],res2.shape[2]),dtype=numpy.float32)
    #edge[0,:-1,:]=(res2[0,:-1]-res2[0,1:])
    #edge[1,:,:-1]=(res2[1,:,:-1]-res2[1,:,1:])
    if mode=="Old":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])+abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[1,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])+abs(res2[1,:,:-1]-res2[1,:,1:])
    elif mode=="New":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])
        edge[1,:,:-1]=abs(res2[1,:,:-1]-res2[1,:,1:])
    elif mode=="Best":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])#V edge Y
        edge[1,:-1,:]=abs(res2[1,:-1,:]-res2[1,1:,:])#V edge X
        edge[2,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])#H edge Y
        edge[3,:,:-1]=abs(res2[1,:,:-1]-res2[1,:,1:])#H edge X
    return dfeat,-edge    

def getfeat_full(m1,pad,res2,movy=None,movx=None,mode="Quad",rot=None):
    if movy==None:
        movy=m1.shape[0]
    if movx==None:
        movx=m1.shape[1]
    pady=(res2.shape[1])*2
    padx=(res2.shape[2])*2
    dfeat=numpy.zeros((res2.shape[1]*2,res2.shape[2]*2,m1.shape[2]),dtype=numpy.float32)
    m1pad=numpy.zeros((m1.shape[0]+2*pady,m1.shape[1]*2+2*padx,m1.shape[2]),dtype=numpy.float32)
    m1pad[pady:m1.shape[0]+pady,padx:m1.shape[1]+padx]=m1
    #m1pad=m1
    for px in range(res2.shape[2]):
        for py in range(res2.shape[1]):
            #dfeat[py*2:py*2+2,px*2:px*2+2]=blk2pad[py*2+res[py,px]/(movx*2+1),px*2+res[py,px]%(movx*2+1)].reshape(2,2,31)
            cpy=py*2+res2[0,py,px]+pady
            cpx=px*2+res2[1,py,px]+padx    
            if rot==None:
                dfeat[py*2:py*2+2,px*2:px*2+2]=m1pad[cpy:cpy+2,cpx:cpx+2]
            else:
                dfeat[py*2:py*2+2,px*2:px*2+2]=rotate(m1pad[cpy:cpy+2,cpx:cpx+2],rot[py,px])
    edge=numpy.zeros((res2.shape[0]*4,res2.shape[1],res2.shape[2]),dtype=numpy.float32)
    #edge[0,:-1,:]=(res2[0,:-1]-res2[0,1:])
    #edge[1,:,:-1]=(res2[1,:,:-1]-res2[1,:,1:])
    if mode=="Old":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])+abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[1,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])+abs(res2[1,:,:-1]-res2[1,:,1:])
    elif mode=="New":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])
        edge[1,:,:-1]=abs(res2[1,:,:-1]-res2[1,:,1:])
    elif mode=="Best":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])
        edge[1,:-1,:]=abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[2,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])
        edge[3,:,:-1]=abs(res2[1,:,:-1]-res2[1,:,1:])
    elif mode=="Quad":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])
        edge[1,:-1,:]=abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[2,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])
        edge[3,:,:-1]=abs(res2[1,:,:-1]-res2[1,:,1:])
        edge[4,:-1,:]=(res2[0,:-1,:]-res2[0,1:,:])**2
        edge[5,:-1,:]=(res2[1,:-1,:]-res2[1,1:,:])**2
        edge[6,:,:-1]=(res2[0,:,:-1]-res2[0,:,1:])**2
        edge[7,:,:-1]=(res2[1,:,:-1]-res2[1,:,1:])**2  
    return dfeat,-edge    

def getfeat_fullN(m1,N,res2,mode="Quad",rot=None):
    movy=m1.shape[0]
    movx=m1.shape[1]
    pady=(res2.shape[1])*N
    padx=(res2.shape[2])*N
    dfeat=numpy.zeros((res2.shape[1]*N,res2.shape[2]*N,m1.shape[2]),dtype=numpy.float32)
    #m1pad=numpy.zeros((m1.shape[0]*N+N*pady,m1.shape[1]*N+N*padx,m1.shape[2]),dtype=numpy.float32)
    m1pad=numpy.zeros((m1.shape[0]*N+2*pady,m1.shape[1]*N+2*padx,m1.shape[2]),dtype=numpy.float32)
    m1pad[pady:m1.shape[0]+pady,padx:m1.shape[1]+padx]=m1
    #m1pad=m1
    for px in range(res2.shape[2]):
        for py in range(res2.shape[1]):
            #dfeat[py*2:py*2+2,px*2:px*2+2]=blk2pad[py*2+res[py,px]/(movx*2+1),px*2+res[py,px]%(movx*2+1)].reshape(2,2,31)
            cpy=py*N+res2[0,py,px]+pady
            cpx=px*N+res2[1,py,px]+padx    
            if rot==None:
                dfeat[py*N:py*N+N,px*N:px*N+N]=m1pad[cpy:cpy+N,cpx:cpx+N]
            else:
                dfeat[py*N:py*N+N,px*N:px*N+N]=rotate(m1pad[cpy:cpy+N,cpx:cpx+N],rot[py,px])
    edge=numpy.zeros((res2.shape[0]*4,res2.shape[1],res2.shape[2]),dtype=numpy.float32)
    #edge[0,:-1,:]=(res2[0,:-1]-res2[0,1:])
    #edge[1,:,:-1]=(res2[1,:,:-1]-res2[1,:,1:])
    if mode=="Old":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])+abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[1,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])+abs(res2[1,:,:-1]-res2[1,:,1:])
    elif mode=="New":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])
        edge[1,:,:-1]=abs(res2[1,:,:-1]-res2[1,:,1:])
    elif mode=="Best":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])
        edge[1,:-1,:]=abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[2,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])
        edge[3,:,:-1]=abs(res2[1,:,:-1]-res2[1,:,1:])
    elif mode=="Quad":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])
        edge[1,:-1,:]=abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[2,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])
        edge[3,:,:-1]=abs(res2[1,:,:-1]-res2[1,:,1:])
        edge[4,:-1,:]=(res2[0,:-1,:]-res2[0,1:,:])**2
        edge[5,:-1,:]=(res2[1,:-1,:]-res2[1,1:,:])**2
        edge[6,:,:-1]=(res2[0,:,:-1]-res2[0,:,1:])**2
        edge[7,:,:-1]=(res2[1,:,:-1]-res2[1,:,1:])**2  
    return dfeat,-edge    


if __name__ == "__main__":

    if 1:
        from pylab import *
        import util
        im=util.myimread("000535.jpg")[:,::-1,:]#flip
        #im=util.myimread("000379.jpg")[:,::-1,:]#flip
        #im=util.myimread("005467.jpg")[:,::-1,:]#flip
        #img=numpy.zeros((100,100,3))
        #subplot(1,2,1)
        imshow(im)
        import pyrHOG2
        f=pyrHOG2.pyrHOG(im,interv=10,savedir="",notload=True,notsave=True,hallucinate=False,cformat=True)

        N=3
        import util
        model1=util.load("./data/bicycle3_bestdef14.model")
        m1=model1[0]["ww"][2]
        aux=numpy.zeros((12,21,31),dtype=numpy.float32)
        aux[:,:20,:]=model1[0]["ww"][-1]
        m1=aux
        import crf3
        numhyp=10
        numy=m1.shape[0]/2
        numx=m1.shape[1]/2
        factor=0.01#0.3
        #mcostm=factor*model1[0]["cost"]
        mcost=factor*numpy.ones((8,numy,numx),dtype=c_float)
        t=time.time()
        ldet=crf3.match_bbN(m1,f.hog,N,mcost,show=False,rot=False,numhyp=120)
        print "Time:",time.time()-t
        rr=[x["scr"] for x in ldet]
        figure(22)
        plot(rr)
        show()
        #fdsfd

    if 1:
        ldet2=[]
        from pylab import *
        import util
        #im=util.myimread("000125.jpg")#flip
        im=util.myimread("000535.jpg")[:,::-1,:]#
        #im=util.myimread("000379.jpg")[:,::-1,:]#flip
        #im=util.myimread("005467.jpg")[:,::-1,:]#flip
        #img=numpy.zeros((100,100,3))
        #subplot(1,2,1)
        imshow(im)
        import pyrHOG2
        f=pyrHOG2.pyrHOG(im,interv=10,savedir="",notload=True,notsave=True,hallucinate=False,cformat=True)

        import util
        #model1=util.load("./data/CRF/12_04_27/bicycle2_NoCRF9.model")
        #model2=util.load("./data/CRF/12_04_27/bicycle2_NoCRFNoDef9.model")
        #model1=util.load("./data/rigid/12_08_17/bicycle3_complete8.model")
        model1=util.load("./data/bicycle3_bestdef14.model")
        m1=model1[0]["ww"][2]
        #make the model to fit for 3x3 parts
        aux=numpy.zeros((12,21,31),dtype=numpy.float32)
        aux[:,:20,:]=model1[0]["ww"][-1]
        m1=aux
        #m1=numpy.tile(m1,(3,3,1))#m1[:m1.shape[0]/2,:m1.shape[1]/2].copy()
        m2=f.hog[28]#[2:18,:24] #12x20 --> padding -> 16x24
        if 0:
            import drawHOG
            img=drawHOG.drawHOG(m1)
            figure(figsize=(15,5))
            subplot(1,2,1)
            title("Model")
            imshow(img)
            img=drawHOG.drawHOG(m2)
            subplot(1,2,2)
            title("HOG image")
            imshow(img)
            raw_input()
        import crf3
        reload(crf3)
        N=3
        numhyp=3
        numy=m1.shape[0]/N
        numx=m1.shape[1]/N
        #movy=(numy*2-1)/2
        #movx=(numx*2-1)/2
        factor=0.01#0.3
        #mcostm=factor*model1[0]["cost"]
        mcostc=factor*numpy.ones((8,numy,numx),dtype=c_float)
        #mcostc=factor*mcostc*numpy.sqrt(numpy.sum(mcostm**2))/numpy.sqrt(numpy.sum(mcostc**2))
        mcost=mcostc
        t=time.time()
        remove=[]
        totqual=0
        col=['w','r','g','b','y','c','k','y','c','k']
        for r in range(len(f.hog)):
            m2=f.hog[r]
            lscr,fres=crf3.match_fullN(m1,m2,N,mcost,show=False,feat=False,rot=False,numhyp=numhyp)
            print "Total time",time.time()-t
            #print "Score",scr
            idraw=False
            if idraw:
                import drawHOG
                #rec=drawHOG.drawHOG(dfeat)
                figure(figsize=(15,5))
                #subplot(1,2,1)
                #imshow(rec)
                title("Reconstructed HOG Image (Learned Costs)")
                subplot(1,2,2)
                img=drawHOG.drawHOG(m2)
            hogpix=15
            myscr=[]
            sf=int(8*2/f.scale[r])
            im2=numpy.zeros((im.shape[0]+sf*numy*2,im.shape[1]+sf*numx*2,im.shape[2]),dtype=im.dtype)
            im2[sf*numy:sf*numy+im.shape[0],sf*numx:sf*numx+im.shape[1]]=im
            for hy in range(fres.shape[0]):
                ldet2.append(lscr[hy])
                res=fres[fres.shape[0]-hy-1]
                dfeat,edge=crf3.getfeat_fullN(m2,N,res)
                print "Scr",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost),"Error",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-lscr[fres.shape[0]-hy-1]
                #print "Edge Lin",numpy.sum(edge[:4]*mcost[:4]),"Error",numpy.sum(m1*dfeat)+numpy.sum(edge[:4]*mcost[:4])-lscr[fres.shape[0]-hy-1]
                #print "Edge Quad",numpy.sum(edge[4:]*mcost[4:]),"Error",numpy.sum(m1*dfeat)+numpy.sum(edge[4:]*mcost[4:])-lscr[fres.shape[0]-hy-1]
                myscr.append(numpy.sum(m1*dfeat)+numpy.sum(edge*mcost))
                rcim=numpy.zeros((sf*numy,sf*numx,3),dtype=im.dtype)
                if idraw:
                    for px in range(res.shape[2]):
                        for py in range(res.shape[1]):
                            util.box(py*N*hogpix+res[0,py,px]*hogpix,px*N*hogpix+res[1,py,px]*hogpix,py*N*hogpix+res[0,py,px]*hogpix+N*hogpix,px*N*hogpix+res[1,py,px]*hogpix+N*hogpix, col=col[fres.shape[0]-hy-1], lw=2)  
                            impy=(py)*sf+(res[0,py,px]+1)*sf/N
                            impx=(px)*sf+(res[1,py,px]+1)*sf/N
                            rcim[sf*py:sf*(py+1),sf*px:sf*(px+1)]=im2[sf*numy+impy:sf*numy+impy+sf,sf*numx+impx:sf*numx+impx+sf] 
                            #m2[py*2+res[0,py,px]:(py+1)*2+res[0,py,px],px*2+res[1,py,px]:(px+1)*2+res[1,py,px]]=0 
                            #text(px*20+(res[py,px]%(movx*2+1)-movx)*10,py*20+(res[py,px]/(movy*2+1)-movy)*10,"(%d,%d)"%(py,px))
                #remove.append(res)
            if idraw:
                imshow(img)
                title("Deformed Grid (Learned Costs)")
                rec=drawHOG.drawHOG(dfeat)
                subplot(1,2,1)
                rec=drawHOG.drawHOG(dfeat)
                imshow(rec)
                figure(15)
                clf()
                imshow(rcim)
                draw()
                show()
                print "QUALITY:",numpy.sum(numpy.array(myscr)*(numpy.arange(numhyp)+1))
                raw_input()
            print "QUALITY:",numpy.sum(numpy.array(myscr)*(numpy.arange(numhyp)+1))
            totqual+=numpy.sum(numpy.array(myscr)*(numpy.arange(numhyp)+1))
            #raw_input()
        print "Total QUALITY:",totqual
        ldet2=-numpy.sort(-numpy.array(ldet2))
        figure(22)
        clf()
        rr=-numpy.sort(-numpy.array(rr))
        plot(rr)
        plot(ldet2,"r")
        show()


    if 0:
        from pylab import *
        import util
        im=util.myimread("000535.jpg")[:,::-1,:]#flip
        #im=util.myimread("000379.jpg")[:,::-1,:]#flip
        #im=util.myimread("005467.jpg")[:,::-1,:]#flip
        #img=numpy.zeros((100,100,3))
        #subplot(1,2,1)
        imshow(im)
        import pyrHOG2
        f=pyrHOG2.pyrHOG(im,interv=10,savedir="",notload=True,notsave=True,hallucinate=False,cformat=True)

        import util
        model1=util.load("./data/bicycle3_bestdef14.model")
        m1=model1[0]["ww"][2]
        #m1=numpy.tile(m1,(3,3,1))#m1[:m1.shape[0]/2,:m1.shape[1]/2].copy()
        m2=f.hog[28]#[2:18,:24] #12x20 --> padding -> 16x24

        import crf3
        numhyp=10
        numy=m1.shape[0]/2
        numx=m1.shape[1]/2
        factor=1.0#0.3
        mcostm=factor*model1[0]["cost"]
        mcostc=numpy.ones((8,numy,numx),dtype=c_float)
        mcostc=factor*mcostc*numpy.sqrt(numpy.sum(mcostm**2))/numpy.sqrt(numpy.sum(mcostc**2))
        mcost=mcostc
        t=time.time()
        ldet=crf3.match_bb(m1,f.hog,mcost,show=False,rot=False,numhyp=120)
        print "Time:",time.time()-t
        rr=[x["scr"] for x in ldet]
        figure(22)
        plot(rr)
        show()
        #fdsfd
        
    if 0:
        ldet2=[]
        from pylab import *
        import util
        #im=util.myimread("000125.jpg")#flip
        im=util.myimread("000535.jpg")[:,::-1,:]#
        #im=util.myimread("000379.jpg")[:,::-1,:]#flip
        #im=util.myimread("005467.jpg")[:,::-1,:]#flip
        #img=numpy.zeros((100,100,3))
        #subplot(1,2,1)
        imshow(im)
        import pyrHOG2
        f=pyrHOG2.pyrHOG(im,interv=10,savedir="",notload=True,notsave=True,hallucinate=False,cformat=True)

        import util
        #model1=util.load("./data/CRF/12_04_27/bicycle2_NoCRF9.model")
        #model2=util.load("./data/CRF/12_04_27/bicycle2_NoCRFNoDef9.model")
        #model1=util.load("./data/rigid/12_08_17/bicycle3_complete8.model")
        model1=util.load("./data/bicycle3_bestdef14.model")
        m1=model1[0]["ww"][2]
        #m1=numpy.tile(m1,(3,3,1))#m1[:m1.shape[0]/2,:m1.shape[1]/2].copy()
        #model1=util.load("./data/CRF/12_09_15/person3_buffy0.model")
        #m1=model1[1]["ww"][0]
        #model1=util.load("../../CFdet/data/CRF/12_08_31/bicycle3_newbiask1011.model")
        #model2=util.load("data/CF/12_08_15/bicycle3_newcache2.model")
        #m2=model2[0]["ww"][2]    
        m2=f.hog[28]#[2:18,:24] #12x20 --> padding -> 16x24
        #m2=f.hog[5]#[15:40,15:40]
        #m2=f.hog[0]
        if 0:
            import drawHOG
            img=drawHOG.drawHOG(m1)
            figure(figsize=(15,5))
            subplot(1,2,1)
            title("Model")
            imshow(img)
            img=drawHOG.drawHOG(m2)
            subplot(1,2,2)
            title("HOG image")
            imshow(img)
        #add paddings
        pad=0
        #m3=numpy.zeros((m2.shape[0]+2*pad,m2.shape[1]+2*pad,m2.shape[2]),dtype=numpy.float32)
        #m3[pad:-pad,pad:-pad]=m2
        #m2=m3

        import crf3
        reload(crf3)
        numhyp=3
        numy=m1.shape[0]/2
        numx=m1.shape[1]/2
        #movy=(numy*2-1)/2
        #movx=(numx*2-1)/2
        factor=1.0#0.3
        mcostm=factor*model1[0]["cost"]
        mcostc=numpy.ones((8,numy,numx),dtype=c_float)
        mcostc=factor*mcostc*numpy.sqrt(numpy.sum(mcostm**2))/numpy.sqrt(numpy.sum(mcostc**2))
        mcost=mcostc
        #mcost=numpy.tile(mcost,(1,2,2))
        #mcost[0,-1,:]=0#vertical 
        #mcost[0,0,:]=0#added v
        #mcost[1,:,-1]=0#horizontal
        #mcost[1,:,0]=0#added o
        t=time.time()
        #scr,res=crf3.match(m1,m2,mcost,pad=pad,show=False,feat=False,rot=False)
        remove=[]
        totqual=0
        col=['w','r','g','b','y','c','k','y','c','k']
        for r in range(len(f.hog)):
            m2=f.hog[r]
            #m2=numpy.zeros((3,3,31),dtype=numpy.float32)
            lscr,fres=crf3.match_full2(m1,m2,mcost,pad=pad,remove=remove,show=False,feat=False,rot=False,numhyp=numhyp)
            print "Total time",time.time()-t
            #print "Score",scr
            idraw=False
            if idraw:
                import drawHOG
                #rec=drawHOG.drawHOG(dfeat)
                figure(figsize=(15,5))
                #subplot(1,2,1)
                #imshow(rec)
                title("Reconstructed HOG Image (Learned Costs)")
                subplot(1,2,2)
                img=drawHOG.drawHOG(m2)
            hogpix=15
            myscr=[]
            sf=int(8*2/f.scale[r])
            im2=numpy.zeros((im.shape[0]+sf*numy*2,im.shape[1]+sf*numx*2,im.shape[2]),dtype=im.dtype)
            im2[sf*numy:sf*numy+im.shape[0],sf*numx:sf*numx+im.shape[1]]=im
            for hy in range(fres.shape[0]):
                ldet2.append(lscr[hy])
                res=fres[fres.shape[0]-hy-1]
                dfeat,edge=crf3.getfeat_full(m2,pad,res)
                print "Scr",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost),"Error",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-lscr[fres.shape[0]-hy-1]
                #print "Edge Lin",numpy.sum(edge[:4]*mcost[:4]),"Error",numpy.sum(m1*dfeat)+numpy.sum(edge[:4]*mcost[:4])-lscr[fres.shape[0]-hy-1]
                #print "Edge Quad",numpy.sum(edge[4:]*mcost[4:]),"Error",numpy.sum(m1*dfeat)+numpy.sum(edge[4:]*mcost[4:])-lscr[fres.shape[0]-hy-1]
                myscr.append(numpy.sum(m1*dfeat)+numpy.sum(edge*mcost))
                rcim=numpy.zeros((sf*numy,sf*numx,3),dtype=im.dtype)
                if idraw:
                    for px in range(res.shape[2]):
                        for py in range(res.shape[1]):
                            util.box(py*2*hogpix+res[0,py,px]*hogpix,px*2*hogpix+res[1,py,px]*hogpix,py*2*hogpix+res[0,py,px]*hogpix+2*hogpix,px*2*hogpix+res[1,py,px]*hogpix+2*hogpix, col=col[fres.shape[0]-hy-1], lw=2)  
                            impy=(py)*sf+(res[0,py,px]+1)*sf/2
                            impx=(px)*sf+(res[1,py,px]+1)*sf/2
                            rcim[sf*py:sf*(py+1),sf*px:sf*(px+1)]=im2[sf*numy+impy:sf*numy+impy+sf,sf*numx+impx:sf*numx+impx+sf] 
                            #m2[py*2+res[0,py,px]:(py+1)*2+res[0,py,px],px*2+res[1,py,px]:(px+1)*2+res[1,py,px]]=0 
                            #text(px*20+(res[py,px]%(movx*2+1)-movx)*10,py*20+(res[py,px]/(movy*2+1)-movy)*10,"(%d,%d)"%(py,px))
                #remove.append(res)
            if idraw:
                imshow(img)
                title("Deformed Grid (Learned Costs)")
                rec=drawHOG.drawHOG(dfeat)
                subplot(1,2,1)
                rec=drawHOG.drawHOG(dfeat)
                imshow(rec)
                figure(15)
                clf()
                imshow(rcim)
                draw()
                show()
                print "QUALITY:",numpy.sum(numpy.array(myscr)*(numpy.arange(numhyp)+1))
                raw_input()
            print "QUALITY:",numpy.sum(numpy.array(myscr)*(numpy.arange(numhyp)+1))
            totqual+=numpy.sum(numpy.array(myscr)*(numpy.arange(numhyp)+1))
            #raw_input()
        print "Total QUALITY:",totqual
        ldet2=-numpy.sort(-numpy.array(ldet2))
        figure(22)
        clf()
        rr=-numpy.sort(-numpy.array(rr))
        plot(rr)
        plot(ldet2,"r")
        show()

    if 0:
        from pylab import *
        import util
        #im=util.myimread("000125.jpg")#flip
        im=util.myimread("000535.jpg")[:,::-1,:]#
        #im=util.myimread("/users/visics/mpederso/databases/VOC2007/VOCdevkit/VOC2007/JPEGImages/000230.jpg")[:,::-1,:]
        #im=util.myimread("/users/visics/mpederso/databases/VOC2007/VOCdevkit/VOC2007/JPEGImages/000038.jpg")[:,::-1,:]
        #im=util.myimread("/users/visics/mpederso/databases/VOC2007/VOCdevkit/VOC2007/JPEGImages/005540.jpg")[:,::-1,:] #besides
        #im=util.myimread("Weiwei_bicycles.jpg")
        #im=util.myimread("407223044_692.jpg")#[:,::-1,:]#
        #im=util.myimread("imges3.jpg")#[:,::-1,:]#
        #im=util.myimread("000379.jpg")[:,::-1,:]#flip
        #subplot(1,2,1)
        imshow(im)
        import pyrHOG2
        f=pyrHOG2.pyrHOG(im,interv=10,savedir="",notload=True,notsave=True,hallucinate=False,cformat=True)

        import util
        #model1=util.load("./data/CRF/12_04_27/bicycle2_NoCRF9.model")
        #model2=util.load("./data/CRF/12_04_27/bicycle2_NoCRFNoDef9.model")
        #model1=util.load("./data/rigid/12_08_17/bicycle3_complete8.model")
        model1=util.load("../../CFdet/data/CRF/12_08_20/bicycle3_bestdef14.model")
        #model1=util.load("../../CFdet/data/CRF/12_08_31/bicycle3_newbiask1011.model")
        #model2=util.load("data/CF/12_08_15/bicycle3_newcache2.model")
        m1=model1[0]["ww"][2]
        #m2=model2[0]["ww"][2]    
        #m2=f.hog[12]#[2:18,:24] #12x20 --> padding -> 16x24
        #m2=f.hog[5]#[15:40,15:40]
        m2=f.hog[0]
        if 0:
            import drawHOG
            img=drawHOG.drawHOG(m1)
            figure(figsize=(15,5))
            subplot(1,2,1)
            title("Model")
            imshow(img)
            img=drawHOG.drawHOG(m2)
            subplot(1,2,2)
            title("HOG image")
            imshow(img)
        #add paddings
        pad=0
        #m3=numpy.zeros((m2.shape[0]+2*pad,m2.shape[1]+2*pad,m2.shape[2]),dtype=numpy.float32)
        #m3[pad:-pad,pad:-pad]=m2
        #m2=m3

        import crf3
        reload(crf3)
        numy=m1.shape[0]/2
        numx=m1.shape[1]/2
        #movy=(numy*2-1)/2
        #movx=(numx*2-1)/2
        factor=0.5
        mcostm=factor*model1[0]["cost"]
        mcostc=numpy.ones((4,numy,numx),dtype=c_float)
        mcostc=factor*mcostc*numpy.sqrt(numpy.sum(mcostm**2))/numpy.sqrt(numpy.sum(mcostc**2))
        mcost=mcostm
        #mcost[0,-1,:]=0#vertical 
        #mcost[0,0,:]=0#added v
        #mcost[1,:,-1]=0#horizontal
        #mcost[1,:,0]=0#added o
        t=time.time()
        #scr,res=crf3.match(m1,m2,mcost,pad=pad,show=False,feat=False,rot=False)
        remove=[]
        for r in range(len(f.hog)-10):
            m2=f.hog[r]
            scr,res=crf3.match_full(m1,m2,mcost,pad=pad,remove=remove,show=False,feat=False,rot=False)
            print "Total time",time.time()-t
            print "Score",scr
            dfeat,edge=crf3.getfeat_full(m2,pad,res)
            print "Error",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr
            if 1:
                import drawHOG
                rec=drawHOG.drawHOG(dfeat)
                figure(figsize=(15,5))
                subplot(1,2,1)
                imshow(rec)
                title("Reconstructed HOG Image (Learned Costs)")
                subplot(1,2,2)
                img=drawHOG.drawHOG(m2)
                hogpix=15
                for px in range(res.shape[2]):
                    for py in range(res.shape[1]):
                        util.box(hogpix*pad+py*2*hogpix+res[0,py,px]*hogpix,hogpix*pad+ px*2*hogpix+res[1,py,px]*hogpix, 
                        hogpix*pad+py*2*hogpix+res[0,py,px]*hogpix+2*hogpix,hogpix*pad+ px*2*hogpix+res[1,py,px]*hogpix+2*hogpix, col='r', lw=2)  
                        #m2[py*2+res[0,py,px]:(py+1)*2+res[0,py,px],px*2+res[1,py,px]:(px+1)*2+res[1,py,px]]=0 
                        #text(px*20+(res[py,px]%(movx*2+1)-movx)*10,py*20+(res[py,px]/(movy*2+1)-movy)*10,"(%d,%d)"%(py,px))
                remove.append(res)
                imshow(img)
                title("Deformed Grid (Learned Costs)")
                show()
                raw_input()

    if 0:
        from pylab import *
        import util
        im=util.myimread("000379.jpg")[:,::-1,:]#flip
        #subplot(1,2,1)
        imshow(im)
        import pyrHOG2
        f=pyrHOG2.pyrHOG(im,interv=5,savedir="",notload=True,notsave=True,hallucinate=False,cformat=True)

        import util
        #model1=util.load("./data/CRF/12_04_27/bicycle2_NoCRF9.model")
        #model2=util.load("./data/CRF/12_04_27/bicycle2_NoCRFNoDef9.model")
        #model1=util.load("./data/rigid/12_08_17/bicycle3_complete8.model")
        model1=util.load("../../CFdet/data/CRF/12_08_20/bicycle3_bestdef14.model")
        #model2=util.load("data/CF/12_08_15/bicycle3_newcache2.model")
        m1=model1[0]["ww"][2]
        #m2=model2[0]["ww"][2]    
        m2=f.hog[6][2:18,:24] #12x20 --> padding -> 16x24
        import drawHOG
        img=drawHOG.drawHOG(m1)
        figure(figsize=(15,5))
        subplot(1,2,1)
        title("Model")
        imshow(img)
        img=drawHOG.drawHOG(m2)
        subplot(1,2,2)
        title("HOG image")
        imshow(img)
        #add paddings
        pad=2
        #m3=numpy.zeros((m2.shape[0]+2*pad,m2.shape[1]+2*pad,m2.shape[2]),dtype=numpy.float32)
        #m3[pad:-pad,pad:-pad]=m2
        #m2=m3

        import crf3
        reload(crf3)
        numy=m1.shape[0]/2
        numx=m1.shape[1]/2
        movy=(numy*2-1)/2
        movx=(numx*2-1)/2
        factor=1
        mcostm=factor*model1[0]["cost"]
        mcostc=numpy.ones((4,numy,numx),dtype=c_float)
        mcostc=factor*mcostc*numpy.sqrt(numpy.sum(mcostm**2))/numpy.sqrt(numpy.sum(mcostc**2))
        mcost=mcostm
        #mcost[0,-1,:]=0#vertical 
        #mcost[0,0,:]=0#added v
        #mcost[1,:,-1]=0#horizontal
        #mcost[1,:,0]=0#added o
        t=time.time()
        scr,res,frot=crf3.match(m1,m2,mcost,movy=m1.shape[0]/4,movx=m1.shape[1]/4,pad=pad,show=False,feat=False,rot=True)
        print "Total time",time.time()-t
        print "Score",scr
        dfeat,edge=crf3.getfeat(m2,pad,res,rot=frot)
        print "Error l",numpy.sum(m1*dfeat)+numpy.sum(edge[:4]*mcost[:4])-scr
        print "Error q",numpy.sum(m1*dfeat)+numpy.sum(edge[4:]*mcost[4:])-scr
        print "Error",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr
        rec=drawHOG.drawHOG(dfeat)
        subplot(1,2,1)
        imshow(rec)
        title("Reconstructed HOG Image (Learned Costs)")
        subplot(1,2,2)
        img=drawHOG.drawHOG(m2)
        hogpix=15
        for px in range(res.shape[2]):
            for py in range(res.shape[1]):
                util.box(hogpix*pad+py*2*hogpix+res[0,py,px]*hogpix,hogpix*pad+ px*2*hogpix+res[1,py,px]*hogpix, 
                hogpix*pad+py*2*hogpix+res[0,py,px]*hogpix+2*hogpix,hogpix*pad+ px*2*hogpix+res[1,py,px]*hogpix+2*hogpix, col='r', lw=2)   
                #text(px*20+(res[py,px]%(movx*2+1)-movx)*10,py*20+(res[py,px]/(movy*2+1)-movy)*10,"(%d,%d)"%(py,px))
        imshow(img)
        title("Deformed Grid (Learned Costs)")

    if 0:#example with HOG
        import util
        #model1=util.load("./data/CRF/12_04_27/bicycle2_NoCRF9.model")
        #model2=util.load("./data/CRF/12_04_27/bicycle2_NoCRFNoDef9.model")
        model1=util.load("./data/rigid/12_08_17/bicycle3_complete8.model")
        model2=util.load("data/CF/12_08_15/bicycle3_newcache2.model")
        m1=model1[0]["ww"][2]
        m2=model2[0]["ww"][2]    
        pad=2
        m3=numpy.zeros((m2.shape[0]+2*pad,m2.shape[1]+2*pad,m2.shape[2]),dtype=numpy.float32)
        m3[pad:-pad,pad:-pad]=m2
        m2=m3

        numy=m1.shape[0]/2
        numx=m1.shape[1]/2
        movy=(numy*2-1)/2
        movx=(numx*2-1)/2
        mcost=0.01*numpy.ones((4,numy,numx),dtype=c_float)
        #mcost[0,-1,:]=0#vertical 
        #mcost[0,0,:]=0#added v
        #mcost[1,:,-1]=0#horizontal
        #mcost[1,:,0]=0#added o
        mcost[0]=mcost[0]*1
        mcost[1]=mcost[1]*1
        #mcost[1,0,:]=1
        #costV=costV(numy,numx,movy,movx,c=0.001,ch=mcost[0],cv=mcost[1])
        t=time.time()
        #scr,res,dfeat,edge=match(m1,m2,mcost,movy=m1.shape[0]/4,movx=m1.shape[1]/4,pad=pad,show=False)
        scr,res=match(m1,m2,mcost,movy=m1.shape[0]/4,movx=m1.shape[1]/4,pad=pad,show=False,feat=False)
        print "Total time",time.time()-t
        print "Score",scr
        dfeat,edge=getfeat(m2,pad,res)
        print "Error",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr
        import pylab
        import drawHOG
        img=drawHOG.drawHOG(dfeat)
        pylab.figure(11);pylab.imshow(img)
        pylab.show()
        #img2=drawHOG.drawHOG(dfeat2)
        #pylab.figure(12);pylab.imshow(img2)


