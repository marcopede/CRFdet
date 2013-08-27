import ctypes
import numpy
from numpy import ctypeslib
from ctypes import c_int,c_double,c_float
import time

ctypes.cdll.LoadLibrary("./libfastDP.so")
lib= ctypes.CDLL("libfastDP.so")

#crfgr(nodes_y,nodes_x,num_pairs,pair,lab_y,lab_x,unary,dist,wcost)
lib.compute_graph.argtypes=[
    c_int,c_int,# num parts
    c_int, #num edges
    numpy.ctypeslib.ndpointer(dtype=c_int,ndim=2,flags="C_CONTIGUOUS"),# edges
    #numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),# costs for each edge
    c_int,# num_lab_y
    c_int,# num_lab_x
    #numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS"),# unary
    #numpy.ctypeslib.ndpointer(dtype=c_float,ndim=4,flags="C_CONTIGUOUS"),# unary
    numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),# unary
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),# dist
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS"),# wcost
    c_int,#num hypotheses
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS"),
    numpy.ctypeslib.ndpointer(dtype=c_int,ndim=3,flags="C_CONTIGUOUS"),#labels
    c_int,#aiter
    c_int#restart
    ]
lib.compute_graph.restype=ctypes.c_float

crfgr=lib.compute_graph

def crfgr2(numy,numx,cost,movy,movx,rdata,numhyp,lscr,res,aiter,restart,check=True,useStar=False):
    if restart>0 or numhyp>1:
        raise Exception("Option not allowed with fastDP")
    numpairs,pairs,unary,wcost=convert_alpha_fastDP(rdata,cost,numy,numx,movy,movx,useStar)
    dist=numpy.zeros((1,1,1),dtype=numpy.float32)#fake argument
    scr=crfgr(numy,numx,pairs.shape[0],pairs,movy,movx,unary,dist,wcost,numhyp,lscr,res,aiter,restart)
    lscr[0]=scr
    #print "FastDP score",scr
    if check:
        un=numpy.sum(rdata[numpy.arange(res.size),res.flatten()])
        #print "unary",un
        pc=0
        resx=res.flatten()
        for l in range(numpairs):
            l0y=resx[pairs[l][0]]/movx;l0x=resx[pairs[l][0]]%movx
            l1y=resx[pairs[l][1]]/movx;l1x=resx[pairs[l][1]]%movx
            pc+=abs(l0y-l1y)*wcost[l,0]+abs(l0x-l1x)*wcost[l,1]+(l0y-l1y)**2*wcost[l,2]+(l0x-l1x)**2*wcost[l,3]
        scr1=-(un+pc)
        lscr[0]=scr1
        scr=scr1
        #print scr
        #print "External score",scr1
            #print "Dist",l0y-l1y,l0x-l1x
            #raw_input()
        if abs(scr-scr1)>0.00001:
            print ".",
            print "FastDP failed! %f Running the slower Alpha expansion"%(scr+(un+pc))
            #import crf3
            #scr=crf3.crfgr2(numy,numx,cost,movy,movx,rdata,numhyp,lscr,res,aiter,restart)
            #print "New score",scr
    return scr

#for multiple hypotheses
def crfgr3(numy,numx,cost,movy,movx,rdata,numhyp,lscr,res,aiter,restart,check=True,useStar=False):
    #if restart>0 or numhyp>1:
    #    raise Exception("Option not allowed with fastDP")
    numpairs,pairs,unary,wcost=convert_alpha_fastDP(rdata,cost,numy,numx,movy,movx,useStar)
    res2=numpy.zeros((2,numy,numx),dtype=c_int)
    dist=numpy.zeros((1,1,1),dtype=numpy.float32)#fake argument
    for h in range(numhyp):
        #if h!=0:
        scr=crfgr(numy,numx,pairs.shape[0],pairs,movy,movx,unary,dist,wcost,1,lscr[h:h+1],res[h:h+1],aiter,restart)
        #else:
        #import crf3
        #rdata=unary.T.copy()
        #scr=crf3.crfgr2(numy,numx,cost,movy,movx,rdata,1,lscr[h:h+1],res[h:h+1],aiter,restart)
        #print "Scr",scr
        res2[0]=(res[h]/(movx))-numy
        res2[1]=(res[h]%(movx))-numx
        #update unary        
        vdata=unary.reshape((movy,movx,numy,numx))
        for py in range(numy):
            for px in range(numx):
                rcy=res2[0,py,px]+numy
                rcx=res2[1,py,px]+numx
                #vdata[rcy-1:rcy+2,rcx-1:rcx+2,py,px]=10
                vdata[max(0,rcy-1):min(rcy+2,movy),max(0,rcx-1):min(rcx+2,movx),py,px]=10 #increased to 10 to avoid repeating detections
                if 0:
                    import pylab
                    pylab.imshow(numpy.sum(numpy.sum(vdata,2),2),interpolation="nearest")
                    pylab.show()
                    raw_input()
        if check:
            un=numpy.sum(rdata[numpy.arange(res[h].size),res[h].flatten()])
            #print "unary",un
            pc=0
            resx=res[h].flatten()
            for l in range(numpairs):
                l0y=resx[pairs[l][0]]/movx;l0x=resx[pairs[l][0]]%movx
                l1y=resx[pairs[l][1]]/movx;l1x=resx[pairs[l][1]]%movx
                pc+=abs(l0y-l1y)*wcost[l,0]+abs(l0x-l1x)*wcost[l,1]+(l0y-l1y)**2*wcost[l,2]+(l0x-l1x)**2*wcost[l,3]
            scr1=-(un+pc)
            lscr[h]=scr1
            scr=scr1
            #print scr
            #print "External score",scr1
            #print "Dist",l0y-l1y,l0x-l1x
            #raw_input()
            if abs(scr-scr1)>0.00001:
                print ".",
                print "FastDP failed! %f Running the slower Alpha expansion"%(scr+(un+pc))
                #import crf3
                #scr=crf3.crfgr2(numy,numx,cost,movy,movx,rdata,numhyp,lscr,res,aiter,restart)
                #print "New score",scr
    scr=lscr[0]
    return scr


def convert_alpha_fastDP(unary,cost,nodes_y,nodes_x,lab_y,lab_x,useStar=False):
    """convert input for fastDP"""
    #useStar=False
    #print "USE star fastDP",useStar
    #dsf
    if useStar:
        #transpose unary terms
        new_unary=unary.T.copy()
        #numhyp=10
        num_lab=lab_x*lab_y
        num_nodes=nodes_y*nodes_x
        num_pairs=num_nodes#nodes_y*nodes_x
        nodes=numpy.arange((nodes_y*nodes_x)).reshape((nodes_y,nodes_x))
        #nodes=numpy.arange((nodes_y*nodes_x)).T.copy().reshape((nodes_y,nodes_x))
        pair=numpy.zeros((num_pairs,2),dtype=numpy.int32)
        wcost=numpy.zeros((num_pairs,4),dtype=numpy.float32)
        cc=0
        #cetner of the star
        cy=nodes_y/2
        cx=nodes_x/2
        for py in range(nodes_y):
            for px in range(nodes_x):
                pair[cc]=[nodes[cy,cx],nodes[py,px]]
                wcost[cc,:4]=cost[:4,py,px]
                cc+=1
        return pair.shape[0],pair,new_unary,wcost
    else:
        #transpose unary terms
        new_unary=unary.T.copy()
        #numhyp=10
        num_lab=lab_x*lab_y
        num_nodes=nodes_y*nodes_x
        num_pairs=(nodes_y-1)*(nodes_x)+(nodes_y)*(nodes_x-1)
        nodes=numpy.arange((nodes_y*nodes_x)).reshape((nodes_y,nodes_x))
        #nodes=numpy.arange((nodes_y*nodes_x)).T.copy().reshape((nodes_y,nodes_x))
        pair=numpy.zeros((num_pairs,2),dtype=numpy.int32)
        wcost=numpy.zeros((num_pairs,4),dtype=numpy.float32)
        cc=0
        #vertical edges
        for py in range(nodes_y-1):
            for px in range(nodes_x):
                pair[cc]=[nodes[py,px],nodes[py+1,px]]
                wcost[cc,:2]=cost[:2,py,px]
                wcost[cc,2:4]=cost[4:6,py,px]
                cc+=1
        #print "Pair 0:",pair[0],"Cost",wcost[0]
        #horizontal edges
        for py in range(nodes_y):
            for px in range(nodes_x-1):
                pair[cc]=[nodes[py,px],nodes[py,px+1]]
                #wcost[cc]=cost[4:,py,px]
                wcost[cc,:2]=cost[2:4,py,px]
                wcost[cc,2:4]=cost[6:8,py,px]
                cc+=1
        #dist=numpy.zeros((num_lab,num_lab,4),dtype=numpy.float32)
        #for l1 in range(num_lab):
        #    for l2 in range(num_lab):
        #        dist[l1,l2]=[abs(l1/lab_x-l2/lab_x),abs(l1%lab_x-l2%lab_x),
        #                     (l1/lab_x-l2/lab_x)**2,(l1%lab_x-l2%lab_x)**2 ]
        return pair.shape[0],pair,new_unary,wcost

if __name__ == "__main__":
    numhyp=10
    nodes_y=10
    nodes_x=5
    num_nodes=nodes_y*nodes_x
    lab_y=20
    lab_x=20
    num_lab=lab_x*lab_y
    num_pairs=(nodes_y-1)*(nodes_x)+(nodes_y)*(nodes_x-1)
    nodes=numpy.arange((nodes_y*nodes_x)).reshape((nodes_y,nodes_x))
    unary=numpy.zeros((lab_y,lab_x,nodes_y,nodes_x),dtype=numpy.float32)
    unary[5,5,:,:]=-1
    unary[6,5,0,0]=-2
    rmin=unary.min()
    rdata=unary-rmin
    #mypair=numpy.zeros((nodes_y,nodes_x,8))
    mypair=0.9*numpy.ones((nodes_y,nodes_x,8))
    pair=numpy.zeros((num_pairs,2),dtype=numpy.int32)
    wcost=numpy.zeros((num_pairs,4),dtype=numpy.float32)
    cc=0
    for py in range(nodes_y):
        for px in range(nodes_x-1):
            pair[cc]=[nodes[py,px],nodes[py,px+1]]
            wcost[cc]=mypair[py,px,:4]
            cc+=1
    for py in range(nodes_y-1):
        for px in range(nodes_x):
            pair[cc]=[nodes[py,px],nodes[py+1,px]]
            wcost[cc]=mypair[py,px,4:]
            cc+=1

    dist=numpy.zeros((num_lab,num_lab,4),dtype=numpy.float32)
    for l1 in range(num_lab):
        for l2 in range(num_lab):
            dist[l1,l2]=[abs(l1/lab_x-l2/lab_x),abs(l1%lab_x-l2%lab_x),
                         (l1/lab_x-l2/lab_x)**2,(l1%lab_x-l2%lab_x)**2 ]

    scrhyp=numpy.zeros((numhyp),dtype=numpy.float32)
    res=numpy.zeros((numhyp,nodes_y,nodes_x),dtype=numpy.int32)
    crfgr(nodes_y,nodes_x,num_pairs,pair,lab_y,lab_x,unary,dist,wcost,numhyp,scrhyp,res,3,0)






