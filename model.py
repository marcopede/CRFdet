
import numpy
#danger: code dupicated in pyrHOG2.py: find a solution


def initmodel(fy,fx,N,useRL,lenf,CRF=False,small2=False):
    ww=[]
    hww=[]
    voc=[]
    dd=[]    
    lev=1
    for l in range(lev):
        if useRL:
            lowf=numpy.zeros((fy*2**l,fx*2**l,lenf)).astype(numpy.float32)
            lowf[:(fy*2**l)/2,:,2]=0.1/lenf
            lowf[(fy*2**l)/2:,:,7]=0.1/lenf
            lowf[:(fy*2**l)/2,:,11]=0.1/lenf
            lowf[(fy*2**l)/2:,:,16]=0.1/lenf
            lowf[:(fy*2**l)/2,:,18+2]=0.1/lenf
            lowf[(fy*2**l)/2:,:,18+7]=0.1/lenf
        else:
            lowf=numpy.ones((fy*2**l,fx*2**l,lenf)).astype(numpy.float32)
        ww.append(lowf)
        rho=0
    mynorm=0
    for wc in ww:
        mynorm+=numpy.sum(wc**2)
    for idw,wc in enumerate(ww):
        ww[idw]=wc*0.1/numpy.sqrt(mynorm)
    model={"ww":ww,"rho":rho,"fy":ww[0].shape[0],"fx":ww[0].shape[1],"N":N}
    if CRF:
        #cost=0.01*numpy.ones((2,fy*2,fx*2),dtype=numpy.float32)
        cost=0.01*numpy.ones((8,fy*2,fx*2),dtype=numpy.float32)
        model["cost"]=cost
    if small2:
        model["small2"]=numpy.array([0.0,0.0,0.0])#2x2,4x4,not used
    model["norm"]=(fy*fx)
    return model

def model2w(model,deform,usemrf,usefather,k=1,lastlev=0,usebow=False,useCRF=False,small2=False):
    w=numpy.zeros(0,dtype=numpy.float32)
    if model.has_key("norm"):
        norm=model["norm"]
    else:
        norm=1
    for l in range(len(model["ww"])-lastlev):
        #print "here"#,item
        w=numpy.concatenate((w,model["ww"][l].flatten()/float(norm)))
    if usebow:
        for l in range(len(model["hist"])-lastlev):
            w=numpy.concatenate((w,model["hist"][l].flatten()))
    if useCRF:
        w=numpy.concatenate((w,(model["cost"]/float(k)).flatten()))
    if small2:
        w=numpy.concatenate((w,model["small2"].flatten()))
    return w

def w2model(descr,N,rho,lev,fsz,fy=[],fx=[],bin=5,siftsize=2,deform=False,usemrf=False,usefather=False,k=1,norm=1,mindef=0.001,useoccl=False,usebow=False,useCRF=False,small2=False):
        #does not work with occlusions
        """
        build a new model from the weights of the SVM
        """     
        ww=[]  
        p=0
        occl=[0]*lev
        d=descr
        for l in range(lev):
            dp=(fy*fx)*4**l*fsz
            ww.append((norm*d[p:p+dp].reshape((fy*2**l,fx*2**l,fsz))).astype(numpy.float32))
            p+=dp
            if useoccl:
                occl[l]=d[p]
                p+=1
        hist=[]
        if usebow:
            for l in range(lev):
                hist.append(d[p:p+bin**(siftsize**2)].astype(numpy.float32))
                #hist.append(numpy.zeros(625,dtype=numpy.float32))
                p=p+bin**(siftsize**2)
        m={"ww":ww,"rho":rho,"fy":fy,"fx":fx,"occl":occl,"N":N}
        if useCRF:
            m["cost"]=((d[p:p+8*(fy/N)*(fx/N)].reshape((8,fy/N,fx/N))*(k)))#.clip(mindef,10))
            p=p+8*(fy/N)*(fx/N)
            #m["cost"]=((d[p:p+4*(2*fy)*(2*fx)].reshape((4,2*fy,2*fx))/float(k)).clip(mindef,10))
            #p=p+4*(2*fy)*(2*fx)
        if small2:
            m["small2"]=d[p:]
        return m


