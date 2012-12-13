import numpy
import ctypes
from ctypes import c_float,c_double,c_int,c_void_p,POINTER,pointer
import pylab

#ctypes.cdll.LoadLibrary("./libfastpegasos.so")
lpeg= ctypes.CDLL("libfastpegasos.so")

lpeg.fast_grad2.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#gr
    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#w
    ,c_int #numcomp
    ,POINTER(c_int) #compx
    ,POINTER(c_int) #compy
    ,POINTER(c_void_p) #ptrsamples
    ,c_int #numsamples
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #label
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #components
    ,c_float #lambda
    ,c_int #iter
    ,c_int #parts
    ,c_int #k
    ,c_int #numthr
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS")#sizereg
    ,c_float #valreg
    ,c_float #lb
    ]
lpeg.fast_grad2.restype=ctypes.c_float

lpeg.fast_grad3.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS")#gr
    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS")#w
    ,c_int #numcomp
    ,POINTER(c_int) #compx
    ,POINTER(c_int) #compy
    ,POINTER(c_void_p) #ptrsamples
    ,c_int #numsamples
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #label
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #components
    ,c_float #lambda
    ,c_int #numthr
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS")#sizereg
    ,c_double #valreg
    ]
lpeg.fast_grad3.restype=ctypes.c_void_p

#void fast_pegasos_comp(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype lambda,int iter,int part)
lpeg.fast_pegasos_comp.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#w
    ,c_int #num comp
    ,POINTER(c_int) #compx
    ,POINTER(c_int) #compy 
    ,POINTER(c_void_p)#samples
    ,c_int #numsamples
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS")#labels
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS")#comp number
    ,c_float #lambda
    ,c_int #iter
    ,c_int #part
    ]

#void fast_pegasos_comp_parall(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype lambda,int iter,int part,int k,int numthr)
lpeg.fast_pegasos_comp_parall.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#w
    ,c_int #numcomp
    ,POINTER(c_int) #compx
    ,POINTER(c_int) #compy
    ,POINTER(c_void_p) #ptrsamples
    ,c_int #numsamples
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #label
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #components
    ,c_float #lambda
    ,c_int #iter
    ,c_int #parts
    ,c_int #k
    ,c_int #numthr
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS")#sizereg
    ,c_float #valreg
    ,c_float #lb
    ]
#fast_obj(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int iter,int part,int k,int numthr,int *sizereg,ftype valreg,ftype lb)

lpeg.fast_obj.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS")#w
    ,c_int #numcomp
    ,POINTER(c_int) #compx
    ,POINTER(c_int) #compy
    ,POINTER(c_void_p) #ptrsamples
    ,c_int #numsamples
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #label
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #components
    ,c_float #lambda
    ,c_int #numthr
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS")#sizereg
    ,c_double #valreg
    ]
lpeg.fast_obj.restype=ctypes.c_double
#bias should be added externaly, it is the last dimonsion and it is not regularized

def objective(trpos,trneg,trposcl,trnegcl,clsize,w,C,sizereg=numpy.zeros(10,dtype=numpy.int32),valreg=0.01):
    print "Sizereg",sizereg
    posloss=0.0
    total=1#float(len(trpos))
    clsum=numpy.concatenate(([0],numpy.cumsum(clsize)))
    hardpos=0.0
    for idl,l in enumerate(trpos):
        c=int(trposcl[idl])
        pstart=clsum[c]
        pend=pstart+clsize[c]
        scr=numpy.sum(w[pstart:pend]*l)#+bias*w[pend-1]
        posloss+=max(0,1-scr)
        if scr<0:
            hardpos+=1
        #print "hinge",max(0,1-scr),"scr",scr
        #raw_input()
    negloss=0
    hardneg=0
    for idl,l in enumerate(trneg):
        c=int(trnegcl[idl])
        pstart=clsum[c]
        pend=pstart+clsize[c]
        scr=numpy.sum(w[pstart:pend]*l)#+bias*w[pend-1])
        negloss+=max(0,1+scr)
        if scr>0:
            hardneg+=1
        #print "hinge",max(0,1+scr),"scr",scr
        #raw_input()
    scr=[]
    for idc in range(len(clsize)):
        pstart=clsum[idc]
        pend=pstart+clsize[idc]
        #scr.append(numpy.sum(w[pstart:pend-1]**2))#skip bias
        scr.append(numpy.sum(w[pstart:pend-1-sizereg[idc]]**2)+numpy.sum((w[pend-sizereg[idc]-1:pend-1]-valreg)**2))    
    #reg=lamda*max(scr)*0.5
    #print "C in OBJECTIVE",C
    reg=(max(scr))*0.5/total
    posloss=C*posloss/total
    negloss=C*negloss/total
    hardpos=C*float(hardpos)/total
    hardneg=C*float(hardneg)/total
    #dsfsd
    return posloss,negloss,reg,(posloss+negloss)+reg,hardpos,hardneg


def objective1(w,trpos,trneg,trposcl,trnegcl,clsize,C,sizereg=numpy.zeros(10,dtype=numpy.int32),valreg=0.01):
    posloss=0.0
    total=1#float(len(trpos))
    clsum=numpy.concatenate(([0],numpy.cumsum(clsize)))
    hardpos=0.0
    for idl,l in enumerate(trpos):
        c=int(trposcl[idl])
        pstart=clsum[c]
        pend=pstart+clsize[c]
        scr=numpy.sum(w[pstart:pend]*l)#+bias*w[pend-1]
        posloss+=max(0,1-scr)
        if scr<0:
            hardpos+=1
        #print "hinge",max(0,1-scr),"scr",scr
        #raw_input()
    negloss=0
    hardneg=0
    for idl,l in enumerate(trneg):
        c=int(trnegcl[idl])
        pstart=clsum[c]
        pend=pstart+clsize[c]
        scr=numpy.sum(w[pstart:pend]*l)#+bias*w[pend-1])
        negloss+=max(0,1+scr)
        if scr>0:
            hardneg+=1
        #print "hinge",max(0,1+scr),"scr",scr
        #raw_input()
    scr=[]
    for idc in range(len(clsize)):
        pstart=clsum[idc]
        pend=pstart+clsize[idc]
        scr.append(numpy.sum(w[pstart:pend-1]**2))#skip bias
    #reg=lamda*max(scr)*0.5
    #print "C in OBJECTIVE",C
    reg=(max(scr))*0.5/total
    posloss=C*posloss/total
    negloss=C*negloss/total
    hardpos=C*float(hardpos)/total
    hardneg=C*float(hardneg)/total
    print "Energy:",(posloss+negloss)+reg
    return (posloss+negloss)+reg

def gradient(w,trpos,trneg,trposcl,trnegcl,clsize,C,sizereg=numpy.zeros(10,dtype=numpy.int32),valreg=0.01):
    #loss
    grad=numpy.zeros(w.shape,dtype=w.dtype)
    total=1#float(len(trpos))
    clsum=numpy.concatenate(([0],numpy.cumsum(clsize)))
    for idl,l in enumerate(trpos):
        c=int(trposcl[idl])
        pstart=clsum[c]
        pend=pstart+clsize[c]
        scr=numpy.sum(w[pstart:pend]*l)#+bias*w[pend-1]
        if scr<1:#produces gradient
            grad[pstart:pend]=grad[pstart:pend]-l
    for idl,l in enumerate(trneg):
        c=int(trnegcl[idl])
        pstart=clsum[c]
        pend=pstart+clsize[c]
        scr=numpy.sum(w[pstart:pend]*l)#+bias*w[pend-1])
        if scr>-1:#produces gradient
            grad[pstart:pend]=grad[pstart:pend]+l
    grad=grad*C
    #reg
    scr=[]
    for idc in range(len(clsize)):
        pstart=clsum[idc]
        pend=pstart+clsize[idc]
        scr.append(numpy.sum(w[pstart:pend-1]**2))#skip bias
    c=numpy.argmax(scr)
    pstart=clsum[c]
    pend=pstart+clsize[c]
    grad[pstart:pend-1]=grad[pstart:pend-1]+w[pstart:pend-1]
    return grad


def trainCompBFGhard(trpos,trneg,fname="",trposcl=None,trnegcl=None,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",mintimes=30,maxtimes=200,eps=0.001,num_stop_count=5,numthr=1,k=1,sizereg=numpy.zeros(10,dtype=numpy.int32),valreg=0.01,lb=0):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    #ff=open(fname,"a")
    if trposcl==None:
        trposcl=numpy.zeros(len(trpos),dtype=numpy.int)
    if trnegcl==None:
        trnegcl=numpy.zeros(len(trneg),dtype=numpy.int)
    numcomp=numpy.array(trposcl).max()+1
    trcomp=[]
    newtrcomp=[]
    trcompcl=[]
    alabel=[]
    label=[]
    for l in range(numcomp):
        trcomp.append([])#*numcomp
        label.append([])
    #trcomp=[trcomp]
    #trcompcl=[]
    #label=[[]]*numcomp
    compx=[0]*numcomp
    compy=[0]*numcomp
    for l in range(numcomp):
        compx[l]=len(trpos[numpy.where(numpy.array(trposcl)==l)[0][0]])#+1
    for p,val in enumerate(trposcl):
        trcomp[val].append(trpos[p].astype(numpy.float64))
        #trcomp[val].append(numpy.concatenate((trpos[p],[bias])).astype(c_float))
        #trcompcl.append(val)
        label[val].append(1)
        compy[val]+=1
    for p,val in enumerate(trnegcl):
        trcomp[val].append(trneg[p].astype(numpy.float64))        
        #trcomp[val].append(numpy.concatenate((trneg[p],[bias])).astype(c_float))        
        #trcompcl.append(val)
        label[val].append(-1)
        compy[val]+=1
    ntimes=len(trpos)+len(trneg)
    fdim=numpy.sum(compx)#len(posnfeat[0])+1
    #bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
    w=numpy.zeros(fdim,dtype=numpy.float64)
    if oldw!=None:
        w=oldw.astype(numpy.float64)
        #w[:-1]=oldw
    #for l in range(posntimes):
    #    bigm[l,:-1]=posnfeat[l]
    #    bigm[l,-1]=bias
    #for l in range(negntimes):
    #    bigm[posntimes+l,:-1]=negnfeat[l]
    #    bigm[posntimes+l,-1]=bias
    #labels=numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes))).astype(numpy.float32)
    for l in range(numcomp):
        trcompcl=numpy.concatenate((trcompcl,numpy.ones(compy[l],dtype=numpy.int32)*l))
        alabel=numpy.concatenate((alabel,numpy.array(label[l])))
    trcompcl=trcompcl.astype(numpy.int32)
    alabel=alabel.astype(numpy.int32)
    arrint=(c_int*numcomp)
    arrfloat=(c_void_p*numcomp)
    #trcomp1=[list()]*numcomp
    for l in range(numcomp):#convert to array
        trcomp[l]=numpy.array(trcomp[l])
        newtrcomp.append(trcomp[l].ctypes.data_as(c_void_p))
    print "Clusters size:",compx
    print "Clusters elements:",compy
    print "Starting Pegasos SVM training"
    obj=0.0
    ncomp=c_int(numcomp)
    
    loss=[]
#####
    bounds=numpy.zeros((2,len(w)),dtype=w.dtype)
    bounds[0]=-1000
    bounds[1]=1000
    pos=0
    for l in range(numcomp):
        bounds[0,pos+compx[l]-sizereg[l]-1:pos+compx[l]-1]=lb
        pos+=compx[l]
    bounds=list(bounds.T)
    
    import scipy.optimize as op
    w,fmin,dd=op.fmin_l_bfgs_b(objective1,w,gradient,(trpos,trneg,trposcl,trnegcl,compx,pc,sizereg,valreg),iprint=0,factr=0.00000001,maxfun=1000,bounds=bounds)#,pgtol=0.00001)
    #w,fmin,dd=op.fmin_l_bfgs_b(objective1,w,approx_grad=True,args=(trpos,trneg,trposcl,trnegcl,compx,pc,sizereg,valreg),iprint=0,factr=1e7,maxfun=1000)
    posl,negl,reg,nobj,hpos,hneg=objective(trpos,trneg,trposcl,trnegcl,compx,w,pc,sizereg,valreg)
    print "Reg",reg,"Loss",posl+negl
    return w.astype(numpy.float32),0,0


def objective_fast(w,trpos,trneg,trposcl,trnegcl,clsize,C,sizereg=numpy.zeros(10,dtype=numpy.int32),valreg=0.01,numthr=4,check=True):
    if check:
        posloss=0.0
        #total=float(len(trpos))
        total=1
        clsum=numpy.concatenate(([0],numpy.cumsum(clsize)))
        hardpos=0.0
        #compute pos loss
        for idl,l in enumerate(trpos):
            c=int(trposcl[idl])
            pstart=clsum[c]
            pend=pstart+clsize[c]
            scr=numpy.sum(w[pstart:pend]*l)#+bias*w[pend-1]
            posloss+=max(0,1-scr)
            if scr<0:
                hardpos+=1
            #print "hinge",max(0,1-scr),"scr",scr
            #raw_input()
        negloss=0
        hardneg=0
        #compute neg loss
        for idl,l in enumerate(trneg):
            c=int(trnegcl[idl])
            pstart=clsum[c]
            pend=pstart+clsize[c]
            scr=numpy.sum(w[pstart:pend]*l)#+bias*w[pend-1])
            negloss+=max(0,1+scr)
            if scr>0:
                hardneg+=1
            #print "hinge",max(0,1+scr),"scr",scr
            #raw_input()
        scr=[]
        for idc in range(len(clsize)):
            pstart=clsum[idc]
            pend=pstart+clsize[idc]
            scr.append(numpy.sum(w[pstart:pend-1]**2))#skip bias
        #reg=lamda*max(scr)*0.5
        #print "C in OBJECTIVE",C
        reg=(max(scr))*0.5/total
        posloss=C*posloss/total
        negloss=C*negloss/total
        hardpos=C*float(hardpos)/total
        hardneg=C*float(hardneg)/total
        #print "Energy:",(posloss+negloss)+reg
    ######
    t=time.time()
    if trposcl==None:
        trposcl=numpy.zeros(len(trpos),dtype=numpy.int)
    if trnegcl==None:
        trnegcl=numpy.zeros(len(trneg),dtype=numpy.int)
    numcomp=numpy.array(trposcl).max()+1
    trcomp=[]
    newtrcomp=[]
    trcompcl=[]
    alabel=[]
    label=[]
    for l in range(numcomp):
        trcomp.append([])#*numcomp
        label.append([])
    compx=[0]*numcomp
    compy=[0]*numcomp
    for l in range(numcomp):
        compx[l]=len(trpos[numpy.where(numpy.array(trposcl)==l)[0][0]])#+1
    for p,val in enumerate(trposcl):
        trcomp[val].append(trpos[p].astype(c_float))
        label[val].append(1)
        compy[val]+=1
    for p,val in enumerate(trnegcl):
        trcomp[val].append(trneg[p].astype(c_float))        
        label[val].append(-1)
        compy[val]+=1
    ntimes=len(trpos)+len(trneg)
    fdim=numpy.sum(compx)#len(posnfeat[0])+1
    #w=numpy.zeros(fdim,dtype=numpy.float32)
    #if oldw!=None:
    #    w=oldw.astype(numpy.float32)
    for l in range(numcomp):
        trcompcl=numpy.concatenate((trcompcl,numpy.ones(compy[l],dtype=numpy.int32)*l))
        alabel=numpy.concatenate((alabel,numpy.array(label[l])))
    trcompcl=trcompcl.astype(numpy.int32)
    alabel=alabel.astype(numpy.int32)
    arrint=(c_int*numcomp)
    arrfloat=(c_void_p*numcomp)
    for l in range(numcomp):#convert to array
        trcomp[l]=numpy.array(trcomp[l])
        newtrcomp.append(trcomp[l].ctypes.data_as(c_void_p))
    #print "Clusters size:",compx
    #print "Clusters elements:",compy
    #print "Starting Pegasos SVM training"
    #obj=0.0
    ncomp=c_int(numcomp)
    #lpeg.fast_grad2(grad1,w.astype(numpy.float32),ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,C,-1,-1,-1,numthr,sizereg,valreg,-1)
    #t=time.time()
    obj=lpeg.fast_obj(w.astype(numpy.float32),ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,C,numthr,sizereg,numpy.float(valreg))
    print "Time:",time.time()-t
    if check:
        obj_slow=(posloss+negloss)+reg
        #print "Loss slow",(posloss+negloss),"Loss",obj
        #print "Reg slow",reg,"Reg",obj
        #print "OBJ slow",obj_slow,"OBJ",obj
        print "OBJslow-OBJ:",obj_slow-obj
        #raw_input()
    #print "Energy",obj
    return float(obj)

def objective_fast2(w,ncomp,compx,compy,newtrcomp,ntimes,alabel,trcompcl,C,numthr,sizereg,valreg):
    obj=lpeg.fast_obj(w,ncomp,compx,compy,newtrcomp,ntimes,alabel,trcompcl,C,numthr,sizereg,valreg)
    #print "Energy",obj
    return float(obj)


import time

def gradient_fast(w,trpos,trneg,trposcl,trnegcl,clsize,C,sizereg=numpy.zeros(10,dtype=numpy.int32),valreg=0.01,numthr=4,check=True):
    #loss
    #print "Gradient Fast!!!"
    if check:
        grad=numpy.zeros(w.shape,dtype=w.dtype)
        total=1#float(len(trpos))
        clsum=numpy.concatenate(([0],numpy.cumsum(clsize)))
        for idl,l in enumerate(trpos):
            c=int(trposcl[idl])
            pstart=clsum[c]
            pend=pstart+clsize[c]
            scr=numpy.sum(w[pstart:pend]*l)#+bias*w[pend-1]
            if scr<1:#produces gradient
                grad[pstart:pend]=grad[pstart:pend]-l
        for idl,l in enumerate(trneg):
            c=int(trnegcl[idl])
            pstart=clsum[c]
            pend=pstart+clsize[c]
            scr=numpy.sum(w[pstart:pend]*l)#+bias*w[pend-1])
            if scr>-1:#produces gradient
                grad[pstart:pend]=grad[pstart:pend]+l
        grad=grad*C
        #reg
        scr=[]
        for idc in range(len(clsize)):
            pstart=clsum[idc]
            pend=pstart+clsize[idc]
            scr.append(numpy.sum(w[pstart:pend-1]**2))#skip bias
        c=numpy.argmax(scr)
        pstart=clsum[c]
        pend=pstart+clsize[c]
        grad[pstart:pend-1]=grad[pstart:pend-1]+w[pstart:pend-1]
    ###################
    t=time.time()
    grad1=numpy.zeros(w.shape,dtype=w.dtype)
    if trposcl==None:
        trposcl=numpy.zeros(len(trpos),dtype=numpy.int)
    if trnegcl==None:
        trnegcl=numpy.zeros(len(trneg),dtype=numpy.int)
    numcomp=numpy.array(trposcl).max()+1
    trcomp=[]
    newtrcomp=[]
    trcompcl=[]
    alabel=[]
    label=[]
    for l in range(numcomp):
        trcomp.append([])#*numcomp
        label.append([])
    compx=[0]*numcomp
    compy=[0]*numcomp
    for l in range(numcomp):
        compx[l]=len(trpos[numpy.where(numpy.array(trposcl)==l)[0][0]])#+1
    for p,val in enumerate(trposcl):
        trcomp[val].append(trpos[p].astype(c_float))
        label[val].append(1)
        compy[val]+=1
    for p,val in enumerate(trnegcl):
        trcomp[val].append(trneg[p].astype(c_float))        
        label[val].append(-1)
        compy[val]+=1
    ntimes=len(trpos)+len(trneg)
    fdim=numpy.sum(compx)#len(posnfeat[0])+1
    #w=numpy.zeros(fdim,dtype=numpy.float32)
    #if oldw!=None:
    #    w=oldw.astype(numpy.float32)
    for l in range(numcomp):
        trcompcl=numpy.concatenate((trcompcl,numpy.ones(compy[l],dtype=numpy.int32)*l))
        alabel=numpy.concatenate((alabel,numpy.array(label[l])))
    trcompcl=trcompcl.astype(numpy.int32)
    alabel=alabel.astype(numpy.int32)
    arrint=(c_int*numcomp)
    arrfloat=(c_void_p*numcomp)
    for l in range(numcomp):#convert to array
        trcomp[l]=numpy.array(trcomp[l])
        newtrcomp.append(trcomp[l].ctypes.data_as(c_void_p))
    #print "Clusters size:",compx
    #print "Clusters elements:",compy
    #print "Starting Pegasos SVM training"
    #obj=0.0
    ncomp=c_int(numcomp)
    #lpeg.fast_grad2(grad1,w.astype(numpy.float32),ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,C,-1,-1,-1,numthr,sizereg,valreg,-1)
    #t=time.time()
    lpeg.fast_grad3(grad1,w,ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,C,numthr,sizereg,numpy.float(valreg))
    print "Time:",time.time()-t
    if check:
        print "Grslow-Gr:",numpy.sum(numpy.abs(grad-grad1))
        assert(numpy.sum(numpy.abs(grad-grad1))<0.0001)
    #raw_input()
    return grad1#.astype(numpy.float)

def gradient_fast2(w,ncomp,compx,compy,newtrcomp,ntimes,alabel,trcompcl,C,numthr,sizereg,valreg):
    grad1=numpy.zeros(w.shape,dtype=w.dtype)
    lpeg.fast_grad3(grad1,w,ncomp,compx,compy,newtrcomp,ntimes,alabel,trcompcl,C,numthr,sizereg,valreg)
    return grad1#.astype(numpy.float)


def trainCompBFG_slow(trpos,trneg,fname="",trposcl=None,trnegcl=None,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",mintimes=30,maxtimes=200,eps=0.001,num_stop_count=5,numthr=1,k=1,sizereg=numpy.zeros(10,dtype=numpy.int32),valreg=0.01,lb=0):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    #ff=open(fname,"a")
    if trposcl==None:
        trposcl=numpy.zeros(len(trpos),dtype=numpy.int)
    if trnegcl==None:
        trnegcl=numpy.zeros(len(trneg),dtype=numpy.int)
    numcomp=numpy.array(trposcl).max()+1
    trcomp=[]
    newtrcomp=[]
    trcompcl=[]
    alabel=[]
    label=[]
    for l in range(numcomp):
        trcomp.append([])#*numcomp
        label.append([])
    #trcomp=[trcomp]
    #trcompcl=[]
    #label=[[]]*numcomp
    compx=[0]*numcomp
    compy=[0]*numcomp
    for l in range(numcomp):
        compx[l]=len(trpos[numpy.where(numpy.array(trposcl)==l)[0][0]])#+1
    for p,val in enumerate(trposcl):
        trcomp[val].append(trpos[p].astype(numpy.float64))
        #trcomp[val].append(numpy.concatenate((trpos[p],[bias])).astype(c_float))
        #trcompcl.append(val)
        label[val].append(1)
        compy[val]+=1
    for p,val in enumerate(trnegcl):
        trcomp[val].append(trneg[p].astype(numpy.float64))        
        #trcomp[val].append(numpy.concatenate((trneg[p],[bias])).astype(c_float))        
        #trcompcl.append(val)
        label[val].append(-1)
        compy[val]+=1
    ntimes=len(trpos)+len(trneg)
    fdim=numpy.sum(compx)#len(posnfeat[0])+1
    #bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
    w=numpy.zeros(fdim,dtype=numpy.float64)
    if oldw!=None:
        w=oldw.astype(numpy.float64)
    for l in range(numcomp):
        trcompcl=numpy.concatenate((trcompcl,numpy.ones(compy[l],dtype=numpy.int32)*l))
        alabel=numpy.concatenate((alabel,numpy.array(label[l])))
    trcompcl=trcompcl.astype(numpy.int32)
    alabel=alabel.astype(numpy.int32)
    arrint=(c_int*numcomp)
    arrfloat=(c_void_p*numcomp)
    #trcomp1=[list()]*numcomp
    for l in range(numcomp):#convert to array
        trcomp[l]=numpy.array(trcomp[l])
        newtrcomp.append(trcomp[l].ctypes.data_as(c_void_p))
    print "Clusters size:",compx
    print "Clusters elements:",compy
    print "Starting Pegasos SVM training"
    obj=0.0
    ncomp=c_int(numcomp)
    
    loss=[]
#####
    bounds=numpy.zeros((2,len(w)),dtype=w.dtype)
    bounds[0]=-1000
    bounds[1]=1000
    pos=0
    for l in range(numcomp):
        bounds[0,pos+compx[l]-sizereg[l]-1:pos+compx[l]-1]=lb
        pos+=compx[l]
    bounds=list(bounds.T)    
    import scipy.optimize as op
    w,fmin,dd=op.fmin_l_bfgs_b(objective_fast,w,gradient_fast,(trpos,trneg,trposcl,trnegcl,compx,pc,sizereg,valreg),iprint=0,factr=0.00000001,maxfun=1000,bounds=bounds)#,pgtol=0.00001)
    #w,fmin,dd=op.fmin_l_bfgs_b(objective1,w,approx_grad=True,args=(trpos,trneg,trposcl,trnegcl,compx,pc,sizereg,valreg),iprint=0,factr=1e7,maxfun=1000)
    posl,negl,reg,nobj,hpos,hneg=objective(trpos,trneg,trposcl,trnegcl,compx,w,pc,sizereg,valreg)
    print "Reg",reg,"Loss",posl+negl
    return w.astype(numpy.float32),0,0

def trainCompBFG(trpos,trneg,fname="",trposcl=None,trnegcl=None,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",mintimes=30,maxtimes=200,eps=0.001,num_stop_count=5,numthr=1,k=1,sizereg=numpy.zeros(10,dtype=numpy.int32),valreg=0.01,lb=0):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    #ff=open(fname,"a")
    if 0:
        if trposcl==None:
            trposcl=numpy.zeros(len(trpos),dtype=numpy.int)
        if trnegcl==None:
            trnegcl=numpy.zeros(len(trneg),dtype=numpy.int)
        numcomp=numpy.array(trposcl).max()+1
        trcomp=[]
        newtrcomp=[]
        trcompcl=[]
        alabel=[]
        label=[]
        for l in range(numcomp):
            trcomp.append([])#*numcomp
            label.append([])
        #trcomp=[trcomp]
        #trcompcl=[]
        #label=[[]]*numcomp
        compx=[0]*numcomp
        compy=[0]*numcomp
        for l in range(numcomp):
            compx[l]=len(trpos[numpy.where(numpy.array(trposcl)==l)[0][0]])#+1
        for p,val in enumerate(trposcl):
            trcomp[val].append(trpos[p].astype(numpy.float64))
            #trcomp[val].append(numpy.concatenate((trpos[p],[bias])).astype(c_float))
            #trcompcl.append(val)
            label[val].append(1)
            compy[val]+=1
        for p,val in enumerate(trnegcl):
            trcomp[val].append(trneg[p].astype(numpy.float64))        
            #trcomp[val].append(numpy.concatenate((trneg[p],[bias])).astype(c_float))        
            #trcompcl.append(val)
            label[val].append(-1)
            compy[val]+=1
        ntimes=len(trpos)+len(trneg)
        fdim=numpy.sum(compx)#len(posnfeat[0])+1
        #bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
        #w=numpy.zeros(fdim,dtype=numpy.float64)
        #if oldw!=None:
        #    w=oldw.astype(numpy.float64)
        for l in range(numcomp):
            trcompcl=numpy.concatenate((trcompcl,numpy.ones(compy[l],dtype=numpy.int32)*l))
            alabel=numpy.concatenate((alabel,numpy.array(label[l])))
        trcompcl=trcompcl.astype(numpy.int32)
        alabel=alabel.astype(numpy.int32)
        arrint=(c_int*numcomp)
        arrfloat=(c_void_p*numcomp)
        #trcomp1=[list()]*numcomp
        for l in range(numcomp):#convert to array
            trcomp[l]=numpy.array(trcomp[l])
            newtrcomp.append(trcomp[l].ctypes.data_as(c_void_p))
        print "Clusters size:",compx
        print "Clusters elements:",compy
        print "Starting Pegasos SVM training"
        obj=0.0
        ncomp=c_int(numcomp)
        loss=[]
#####
#    t=time.time()
#    grad1=numpy.zeros(w.shape,dtype=w.dtype)
    if trposcl==None:
        trposcl=numpy.zeros(len(trpos),dtype=numpy.int)
    if trnegcl==None:
        trnegcl=numpy.zeros(len(trneg),dtype=numpy.int)
    numcomp=numpy.array(trposcl).max()+1
    trcomp=[]
    newtrcomp=[]
    trcompcl=[]
    alabel=[]
    label=[]
    for l in range(numcomp):
        trcomp.append([])#*numcomp
        label.append([])
    compx=[0]*numcomp
    compy=[0]*numcomp
    for l in range(numcomp):
        compx[l]=len(trpos[numpy.where(numpy.array(trposcl)==l)[0][0]])#+1
    for p,val in enumerate(trposcl):
        trcomp[val].append(trpos[p].astype(c_float))
        label[val].append(1)
        compy[val]+=1
    for p,val in enumerate(trnegcl):
        trcomp[val].append(trneg[p].astype(c_float))        
        label[val].append(-1)
        compy[val]+=1
    ntimes=len(trpos)+len(trneg)
    fdim=numpy.sum(compx)#len(posnfeat[0])+1
    w=numpy.zeros(fdim,dtype=float)
    if oldw!=None:
        w=oldw.astype(float)
    for l in range(numcomp):
        trcompcl=numpy.concatenate((trcompcl,numpy.ones(compy[l],dtype=numpy.int32)*l))
        alabel=numpy.concatenate((alabel,numpy.array(label[l])))
    trcompcl=trcompcl.astype(numpy.int32)
    alabel=alabel.astype(numpy.int32)
    arrint=(c_int*numcomp)
    arrfloat=(c_void_p*numcomp)
    for l in range(numcomp):#convert to array
        trcomp[l]=numpy.array(trcomp[l])
        newtrcomp.append(trcomp[l].ctypes.data_as(c_void_p))
    #print "Clusters size:",compx
    #print "Clusters elements:",compy
    #print "Starting Pegasos SVM training"
    #obj=0.0
    ncomp=c_int(numcomp)
    bounds=numpy.zeros((2,len(w)),dtype=w.dtype)
    bounds[0]=-1000
    bounds[1]=1000
    pos=0
    for l in range(numcomp):
        bounds[0,pos+compx[l]-sizereg[l]-1:pos+compx[l]-1]=lb
        pos+=compx[l]
    bounds=list(bounds.T)  
    C=pc  
    import scipy.optimize as op
    w,fmin,dd=op.fmin_l_bfgs_b(objective_fast2,w,gradient_fast2,(ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,C,numthr,sizereg,numpy.float(valreg)),disp=1,m=100,factr=10000,maxfun=2000,bounds=bounds,pgtol=0.000001)
    #w,fmin,dd=op.fmin_l_bfgs_b(objective1,w,approx_grad=True,args=(trpos,trneg,trposcl,trnegcl,compx,pc,sizereg,valreg),iprint=0,factr=1e7,maxfun=1000)
    posl,negl,reg,nobj,hpos,hneg=objective(trpos,trneg,trposcl,trnegcl,compx,w,pc,sizereg,valreg)
    print "Reg",reg,"Loss",posl+negl
    return w.astype(numpy.float32),0,0


def trainCompBFG_right(trpos,trneg,fname="",trposcl=None,trnegcl=None,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",mintimes=30,maxtimes=200,eps=0.001,num_stop_count=5,numthr=1,k=1,sizereg=numpy.zeros(10,dtype=numpy.int32),valreg=0.01,lb=0):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    #ff=open(fname,"a")
    if trposcl==None:
        trposcl=numpy.zeros(len(trpos),dtype=numpy.int)
    if trnegcl==None:
        trnegcl=numpy.zeros(len(trneg),dtype=numpy.int)
    numcomp=numpy.array(trposcl).max()+1
    trcomp=[]
    newtrcomp=[]
    trcompcl=[]
    alabel=[]
    label=[]
    for l in range(numcomp):
        trcomp.append([])#*numcomp
        label.append([])
    #trcomp=[trcomp]
    #trcompcl=[]
    #label=[[]]*numcomp
    compx=[0]*numcomp
    compy=[0]*numcomp
    for l in range(numcomp):
        compx[l]=len(trpos[numpy.where(numpy.array(trposcl)==l)[0][0]])#+1
    for p,val in enumerate(trposcl):
        trcomp[val].append(trpos[p].astype(numpy.float64))
        #trcomp[val].append(numpy.concatenate((trpos[p],[bias])).astype(c_float))
        #trcompcl.append(val)
        label[val].append(1)
        compy[val]+=1
    for p,val in enumerate(trnegcl):
        trcomp[val].append(trneg[p].astype(numpy.float64))        
        #trcomp[val].append(numpy.concatenate((trneg[p],[bias])).astype(c_float))        
        #trcompcl.append(val)
        label[val].append(-1)
        compy[val]+=1
    ntimes=len(trpos)+len(trneg)
    fdim=numpy.sum(compx)#len(posnfeat[0])+1
    #bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
    w=numpy.zeros(fdim,dtype=numpy.float64)
    if oldw!=None:
        w=oldw.astype(numpy.float64)
        #w[:-1]=oldw
    #for l in range(posntimes):
    #    bigm[l,:-1]=posnfeat[l]
    #    bigm[l,-1]=bias
    #for l in range(negntimes):
    #    bigm[posntimes+l,:-1]=negnfeat[l]
    #    bigm[posntimes+l,-1]=bias
    #labels=numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes))).astype(numpy.float32)
    for l in range(numcomp):
        trcompcl=numpy.concatenate((trcompcl,numpy.ones(compy[l],dtype=numpy.int32)*l))
        alabel=numpy.concatenate((alabel,numpy.array(label[l])))
    trcompcl=trcompcl.astype(numpy.int32)
    alabel=alabel.astype(numpy.int32)
    arrint=(c_int*numcomp)
    arrfloat=(c_void_p*numcomp)
    #trcomp1=[list()]*numcomp
    for l in range(numcomp):#convert to array
        trcomp[l]=numpy.array(trcomp[l])
        newtrcomp.append(trcomp[l].ctypes.data_as(c_void_p))
    print "Clusters size:",compx
    print "Clusters elements:",compy
    print "Starting Pegasos SVM training"
    obj=0.0
    ncomp=c_int(numcomp)
    
    loss=[]
#####
    bounds=numpy.zeros((2,len(w)),dtype=w.dtype)
    bounds[0]=-1000
    bounds[1]=1000
    pos=0
    for l in range(numcomp):
        bounds[0,pos+compx[l]-sizereg[l]-1:pos+compx[l]-1]=lb
        pos+=compx[l]
    bounds=list(bounds.T)    
    import scipy.optimize as op
    w,fmin,dd=op.fmin_l_bfgs_b(objective1,w,gradient,(trpos,trneg,trposcl,trnegcl,compx,pc,sizereg,valreg),iprint=0,factr=10000000,maxfun=1000,bounds=bounds,pgtol=0.00001,iptrint=0)
    #w,fmin,dd=op.fmin_l_bfgs_b(objective1,w,approx_grad=True,args=(trpos,trneg,trposcl,trnegcl,compx,pc,sizereg,valreg),iprint=0,factr=1e7,maxfun=1000)
    posl,negl,reg,nobj,hpos,hneg=objective(trpos,trneg,trposcl,trnegcl,compx,w,pc,sizereg,valreg)
    print "Reg",reg,"Loss",posl+negl
    return w.astype(numpy.float32),0,0



def trainCompSGD(trpos,trneg,fname="",trposcl=None,trnegcl=None,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",mintimes=30,maxtimes=200,eps=0.001,num_stop_count=5,numthr=1,k=1,sizereg=numpy.zeros(10,dtype=numpy.int32),valreg=0.01,lb=0):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    #ff=open(fname,"a")
    if trposcl==None:
        trposcl=numpy.zeros(len(trpos),dtype=numpy.int)
    if trnegcl==None:
        trnegcl=numpy.zeros(len(trneg),dtype=numpy.int)
    numcomp=numpy.array(trposcl).max()+1
    trcomp=[]
    newtrcomp=[]
    trcompcl=[]
    alabel=[]
    label=[]
    for l in range(numcomp):
        trcomp.append([])#*numcomp
        label.append([])
    #trcomp=[trcomp]
    #trcompcl=[]
    #label=[[]]*numcomp
    compx=[0]*numcomp
    compy=[0]*numcomp
    for l in range(numcomp):
        compx[l]=len(trpos[numpy.where(numpy.array(trposcl)==l)[0][0]])#+1
    for p,val in enumerate(trposcl):
        trcomp[val].append(trpos[p].astype(c_float))
        #trcomp[val].append(numpy.concatenate((trpos[p],[bias])).astype(c_float))
        #trcompcl.append(val)
        label[val].append(1)
        compy[val]+=1
    for p,val in enumerate(trnegcl):
        trcomp[val].append(trneg[p].astype(c_float))        
        #trcomp[val].append(numpy.concatenate((trneg[p],[bias])).astype(c_float))        
        #trcompcl.append(val)
        label[val].append(-1)
        compy[val]+=1
    ntimes=len(trpos)+len(trneg)
    fdim=numpy.sum(compx)#len(posnfeat[0])+1
    #bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
    w=numpy.zeros(fdim,dtype=numpy.float32)
    if oldw!=None:
        w=oldw.astype(numpy.float32)
        #w[:-1]=oldw
    #for l in range(posntimes):
    #    bigm[l,:-1]=posnfeat[l]
    #    bigm[l,-1]=bias
    #for l in range(negntimes):
    #    bigm[posntimes+l,:-1]=negnfeat[l]
    #    bigm[posntimes+l,-1]=bias
    #labels=numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes))).astype(numpy.float32)
    for l in range(numcomp):
        trcompcl=numpy.concatenate((trcompcl,numpy.ones(compy[l],dtype=numpy.int32)*l))
        alabel=numpy.concatenate((alabel,numpy.array(label[l])))
    trcompcl=trcompcl.astype(numpy.int32)
    alabel=alabel.astype(numpy.int32)
    arrint=(c_int*numcomp)
    arrfloat=(c_void_p*numcomp)
    #trcomp1=[list()]*numcomp
    for l in range(numcomp):#convert to array
        trcomp[l]=numpy.array(trcomp[l])
        newtrcomp.append(trcomp[l].ctypes.data_as(c_void_p))
    print "Clusters size:",compx
    print "Clusters elements:",compy
    print "Starting Pegasos SVM training"
    obj=0.0
    ncomp=c_int(numcomp)
    
    loss=[]
    posl,negl,reg,nobj,hpos,hneg=objective(trpos,trneg,trposcl,trnegcl,compx,w,pc,sizereg,valreg)
    loss.append([posl,negl,reg,nobj,hpos,hneg])
    for tt in range(maxtimes):
        lpeg.fast_pegasos_comp_parall(w,ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,int(ntimes*10.0/float(k)),tt+10,k,numthr,sizereg,valreg,lb)#added tt+10 to not restart form scratch
        posl,negl,reg,nobj,hpos,hneg=objective(trpos,trneg,trposcl,trnegcl,compx,w,pc,sizereg,valreg)
        loss.append([posl,negl,reg,nobj,hpos,hneg])
        print "Objective Function:",nobj
        print "PosLoss:%.6f NegLoss:%.6f Reg:%.6f"%(posl,negl,reg)
        #ff.write("Objective Function:%f\n"%nobj)
        ratio=abs(abs(obj/nobj)-1)
        print "Ratio:",ratio
        #ff.write("Ratio:%f\n"%ratio)
        if ratio<eps and tt>mintimes:
            if stop_count==0:
                print "Converging after %d iterations"%tt
                break
            else:
                print "Missing ",stop_count," iterations to converge" 
                stop_count-=1
        else:
            stop_count=num_stop_count
        obj=nobj
    return w,0,loss



