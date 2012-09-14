#some additional functions

import numpy

def getfeat(a,y1,y2,x1,x2,trunc=0):
    """
        returns the hog features at the given position and 
        zeros in case the coordiantes are outside the borders
        """
    dimy=a.shape[0]
    dimx=a.shape[1]
    py1=y1;py2=y2;px1=x1;px2=x2
    dy1=0;dy2=0;dx1=0;dx2=0
    #if trunc>0:
    b=numpy.zeros((abs(y2-y1),abs(x2-x1),a.shape[2]+trunc),dtype=a.dtype)
    if trunc>0:
        b[:,:,-trunc]=1
    #else:
    #    b=numpy.zeros((abs(y2-y1),abs(x2-x1),a.shape[2]))
    if py1<0:
        py1=0
        dy1=py1-y1
    if py2>=dimy:
        py2=dimy
        dy2=y2-py2
    if px1<0:
        px1=0
        dx1=px1-x1
    if px2>=dimx:
        px2=dimx
        dx2=x2-px2
    if numpy.array(a[py1:py2,px1:px2].shape).min()==0 or numpy.array(b[dy1:y2-y1-dy2,dx1:x2-x1-dx2].shape).min()==0:
        pass
    else:
        if trunc==1:
            b[dy1:y2-y1-dy2,dx1:x2-x1-dx2,:-1]=a[py1:py2,px1:px2]
            #b[:,:,-1]=1
            b[dy1:y2-y1-dy2,dx1:x2-x1-dx2,-1]=0
        else:
            b[dy1:y2-y1-dy2,dx1:x2-x1-dx2]=a[py1:py2,px1:px2]
    return b

def myzoom(img,factor,order):
    auxf=numpy.array(factor)
    order=1 #force order to be 1 otherwise I do not know if it is still correct
    from scipy.ndimage import zoom
    aux=img.copy()
    while (auxf[0]<0.5 and auxf[1]<0.5):
        aux=zoom(aux,(0.5,0.5,1),order=order)
        auxf[0]=auxf[0]*2
        auxf[1]=auxf[1]*2
    aux=zoom(aux,auxf,order=order)
    return aux





