import numpy
import ctypes
from ctypes import c_float,c_int,c_void_p,POINTER,pointer
import pylab

ctypes.cdll.LoadLibrary("./libfastpegasos.so")
lpeg= ctypes.CDLL("libfastpegasos.so")

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

#bias should be added externaly, it is the last dimonsion and it is not regularized

def objective(trpos,trneg,trposcl,trnegcl,clsize,w,C,sizereg=numpy.zeros(10,dtype=numpy.int32),valreg=0.01):
    posloss=0.0
    total=float(len(trpos))
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
    #dsfsd
    return posloss,negloss,reg,(posloss+negloss)+reg,hardpos,hardneg


def objective1(w,trpos,trneg,trposcl,trnegcl,clsize,C,sizereg=numpy.zeros(10,dtype=numpy.int32),valreg=0.01):
    posloss=0.0
    total=float(len(trpos))
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

def objectiveSM(w,trpos,trneg,trposcl,trnegcl,clsize,C,sizereg=numpy.zeros(10,dtype=numpy.int32),valreg=0.01):
    posloss=0.0
    total=float(len(trpos))
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
    return (posloss+negloss)+reg


def gradient(w,trpos,trneg,trposcl,trnegcl,clsize,C,sizereg=numpy.zeros(10,dtype=numpy.int32),valreg=0.01):
    #loss
    grad=numpy.zeros(w.shape,dtype=w.dtype)
    total=float(len(trpos))
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

def trainCompBFG(trpos,trneg,fname="",trposcl=None,trnegcl=None,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",mintimes=30,maxtimes=200,eps=0.001,num_stop_count=5,numthr=1,k=1,sizereg=numpy.zeros(10,dtype=numpy.int32),valreg=0.01,lb=0):
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



