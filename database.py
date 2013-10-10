import numpy
import pylab
try:
    import scipy.misc.pilutil as pil
except:
    import scipy.misc as pil
import string
import pickle

from util import myimread
#def myimread(imgname):
#    img=None
#    if imgname.split(".")[-1]=="png":
#        img=pylab.imread(imgname)
#    else:
#        img=pil.imread(imgname)        
#    if img.ndim<3:
#        aux=numpy.zeros((img.shape[0],img.shape[1],3))
#        aux[:,:,0]=img
#        aux[:,:,1]=img
#        aux[:,:,2]=img
#        img=aux
#    return img


def getbboxINRIA(filename):
    """
    get the ground truth bbox from the INRIA database at filename
    """
    fd=open(filename,"r")
    lines=fd.readlines()
    rect=[]
    for idx,item in enumerate(lines):
        p=item.find("Bounding box")
        if p!=-1:
            p=item.find("PASperson")
            if p!=-1:
                p=item.find(":")
                item=item[p:]
                #print item[p:]
                p=item.find("(")
                pXmin=int(item[p+1:].split(" ")[0][:-1])
                pYmin=int(item[p+1:].split(" ")[1][:-1])
                p=item[p:].find("-")
                item=item[p:]
                p=item.find("(")
                pXmax=int(item[p+1:].split(" ")[0][:-1])
                pYmax=int(item[p+1:].split(" ")[1][:-2])
                rect.append((pYmin,pXmin,pYmax,pXmax,0,0))
    return rect

def getbboxVOC06(filename,cl="person",usetr=False,usedf=False):
    """
    get the ground truth bbox from the PASCAL VOC 2006 database at filename
    """
    fd=open(filename,"r")
    lines=fd.readlines()
    rect=[]
    cl="PAS"+cl
    for idx,item in enumerate(lines):
        p=item.find("Bounding box")#look for the bounding box
        if p!=-1:
            p=item.find(cl)#check if it is a person
            if p!=-1:
                p=item.find("Difficult")#check that it is not truncated
                if p==-1 or usedf:
                    p=item.find("Trunc")#check that it is not truncated
                    if p==-1 or usetr:
                        p=item.find(":")
                        item=item[p:]
                        #print item[p:]
                        p=item.find("(")
                        pXmin=int(item[p+1:].split(" ")[0][:-1])
                        pYmin=int(item[p+1:].split(" ")[1][:-1])
                        p=item[p:].find("-")
                        item=item[p:]
                        p=item.find("(")
                        pXmax=int(item[p+1:].split(" ")[0][:-1])
                        pYmax=int(item[p+1:].split(" ")[1][:-3])
                        rect.append((pYmin,pXmin,pYmax,pXmax,0,0))
    return rect

import xml.dom.minidom
from xml.dom.minidom import Node

def getbboxVOC07(filename,cl="person",usetr=False,usedf=False):
    """
    get the ground truth bbox from the PASCAL VOC 2007 database at filename
    """
    rect=[]
    doc = xml.dom.minidom.parse(filename)
    for node in doc.getElementsByTagName("object"):
        #print node
        tr=0
        df=0
        if node.getElementsByTagName("name")[0].childNodes[0].data==cl:
            pose=node.getElementsByTagName("pose")[0].childNodes[0].data#last
            if node.getElementsByTagName("difficult")[0].childNodes[0].data=="0" or usedf:
                if node.getElementsByTagName("truncated")[0].childNodes[0].data=="0" or usetr:
                    if node.getElementsByTagName("difficult")[0].childNodes[0].data=="1":
                        df=1
                    if node.getElementsByTagName("truncated")[0].childNodes[0].data=="1":
                        tr=1
                    l=node.getElementsByTagName("bndbox")
                    #print l
                    for el in l:
                        #print el.parentNode.nodeName
                        if el.parentNode.nodeName=="object":
                            xmin=int(el.getElementsByTagName("xmin")[0].childNodes[0].data)
                            ymin=int(el.getElementsByTagName("ymin")[0].childNodes[0].data)
                            xmax=int(el.getElementsByTagName("xmax")[0].childNodes[0].data)
                            ymax=int(el.getElementsByTagName("ymax")[0].childNodes[0].data)
                            #rect.append((ymin,xmin,ymax,xmax,tr,df))
                            rect.append((ymin,xmin,ymax,xmax,tr,df,pose))#last
    return rect

class imageData:
    """
    interface call to handle a database
    """
    def __init__():
        print "Not implemented"
        
    def getDBname():
        return "Not implemented"
        
    def getImage(i):
        """
        gives the ith image from the database
        """
        print "Not implemented"
    
    def getImageName(i): 
        """
        gives the ith image name from the database
        """
        print "Not implemented"
        
    def getBBox(self,i):
        """
        retrun a list of ground truth bboxs from the ith image
        """
        #print "Not implemented"
        return []
        
    def getTotal():
        """
         return the total number of images in the db
        """
        print "Not implemented"
    
def getRecord(data,total=-1,pos=True,pose=False,facial=False):
    """return all the gt data in a record"""
    if total==-1:
        total=data.getTotal()
    else:
        total=min(data.getTotal(),total)
    if facial:
        arrPos=numpy.zeros(total,dtype=[("id",numpy.int32),("name",object),("bbox",list),("facial",object)])
        if pose:
            arrPos=numpy.zeros(total,dtype=[("id",numpy.int32),("name",object),("bbox",list),("facial",object),("pose",object)])
    else:
        arrPos=numpy.zeros(total,dtype=[("id",numpy.int32),("name",object),("bbox",list)])
    for i in range(total):
        arrPos[i]["id"]=i
        arrPos[i]["name"]=data.getImageName(i)
        arrPos[i]["bbox"]=data.getBBox(i)
        if pose:
            arrPos[i]["pose"]=data.getPose(i)
        if facial:
            arrPos[i]["facial"]=data.getFacial(i)
    return arrPos


class InriaPosData(imageData):
    """
    INRIA database for positive examples
    """
    def __init__(self,basepath="/home/databases/",
                        trainfile="INRIAPerson/Train/pos.lst",
                        imagepath="INRIAPerson/Train/pos/",
                        local="INRIAPerson/",
                        annpath="INRIAPerson/Train/annotations/"):
        self.basepath=basepath        
        self.trainfile=basepath+trainfile
        self.imagepath=basepath+imagepath
        self.annpath=basepath+annpath
        self.local=basepath+local
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        fd.close()
        
    def getStorageDir(self):
        return self.local

    def getDBname(self):
        return "INRIA POS"

    def getImageByName(self,name):
        return myimread(name)
        
    def getImage(self,i):
        item=self.trlines[i]
        return myimread((self.imagepath+item.split("/")[-1])[:-1])
    
    def getImageName(self,i):
        item=self.trlines[i]
        return (self.imagepath+item.split("/")[-1])[:-1]
    
    def getBBox(self,i,cl="",usetr="",usedf=""):
        item=self.trlines[i]
        filename=self.annpath+item.split("/")[-1].split(".")[0]+".txt"
        return getbboxINRIA(filename)
    
    def getTotal(self):
        return len(self.trlines)

#class InriaParts(imageData):
#    """
#    INRIA database for positive examples
#    """
#    def __init__(self,numparts,part,select="pos",basepath="/home/databases/",
#                        trainfile="INRIAPerson/Train/pos.lst",
#                        imagepath="INRIAPerson/Train/pos/",
#                        local="INRIAPerson/",
#                        annpath="INRIAPerson/Train/annotations/"):
#        self.basepath=basepath
#        self.select=select
#        if select=="pos":
#            self.trainfile=basepath+trainfile
#            self.imagepath=basepath+imagepath
#        else:
#            self.trainfile=basepath+"INRIAPerson/Train/neg.lst"
#            self.imagepath=basepath+"INRIAPerson/Train/neg/"
#        self.annpath=basepath+annpath
#        self.local=basepath+local
#        fd=open(self.trainfile,"r")
#        self.trlines=fd.readlines()
#        self.part=part
#        self.numparts=numparts
#        fd.close()
#        
#    def getStorageDir(self):
#        return self.local

#    def getDBname(self):
#        return "INRIA POS"

#    def getImageByName(self,name):
#        return myimread(name)
#        
#    def getImage(self,i):
#        item=self.trlines[i*self.numparts+self.part]
#        return myimread((self.imagepath+item.split("/")[-1])[:-1])
#    
#    def getImageName(self,i):
#        item=self.trlines[i*self.numparts+self.part]
#        return (self.imagepath+item.split("/")[-1])[:-1]
#    
#    def getBBox(self,i,cl="",usetr="",usedf=""):
#        if self.select=="neg":
#            return []
#        item=self.trlines[i*self.numparts+self.part]
#        filename=self.annpath+item.split("/")[-1].split(".")[0]+".txt"
#        return getbboxINRIA(filename)
#    
#    def getTotal(self):
#        return len(self.trlines[self.part::self.numparts])

class InriaNegData(imageData):
    """
        INRIA database for negative examples (no bbox method)
    """
    def __init__(self,basepath="/home/databases/",
                        trainfile="INRIAPerson/Train/neg.lst",
                        imagepath="INRIAPerson/Train/neg/",
                        local="INRIAPerson/",
                        ):
        self.basepath=basepath
        self.trainfile=basepath+trainfile
        self.imagepath=basepath+imagepath
        self.local=basepath+local
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()[::-1]#to take first the big images
        fd.close()
        
    def getDBname():
        return "INRIA NEG"

    def getStorageDir(self):
        return self.local#basepath+"INRIAPerson/"
        
    def getImage(self,i):
        item=self.trlines[i]
        return myimread((self.imagepath+item.split("/")[-1])[:-1])
    
    def getImageName(self,i):
        item=self.trlines[i]
        return (self.imagepath+item.split("/")[-1])[:-1]
    
    def getTotal(self):
        return len(self.trlines)
    
class InriaTestData(imageData):#not done yet
    """
    INRIA database for positive examples
    """
    def __init__(self,basepath="/home/databases/",
                        trainfile="INRIAPerson/Test/pos.lst",
                        imagepath="INRIAPerson/Test/pos/",
                        local="INRIAPerson/",
                        annpath="INRIAPerson/Test/annotations/"):
        self.basepath=basepath
        self.trainfile=basepath+trainfile
        self.imagepath=basepath+imagepath
        self.annpath=basepath+annpath
        self.local=basepath+local
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        fd.close()
        
    def getDBname():
        return "INRIA POS"

    def getStorageDir(self):
        return self.local
        
    def getImage(self,i):
        item=self.trlines[i]
        return myimread((self.imagepath+item.split("/")[-1])[:-1])
    
    def getImageName(self,i):
        item=self.trlines[i]
        return (self.imagepath+item.split("/")[-1])[:-1]
    
    def getBBox(self,i):
        item=self.trlines[i]
        filename=self.annpath+item.split("/")[-1].split(".")[0]+".txt"
        return getbboxINRIA(filename)
    
    def getTotal(self):
        return len(self.trlines)

class InriaTestFullData(imageData):
    """
    INRIA database for positive examples
    """
    def __init__(self,basepath="/home/databases/",
                        trainfile="INRIAPerson/Test/pos.lst",
                        imagepath="INRIAPerson/Test/pos/",
                        imagepath2="INRIAPerson/Test/neg/",
                        local="INRIAPerson/",
                        annpath="INRIAPerson/Test/annotations/"):
        self.basepath=basepath
        self.trainfile=basepath+trainfile
        self.imagepath=basepath+imagepath
        self.imagepath2=basepath+imagepath2
        self.annpath=basepath+annpath
        self.local=basepath+local
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        self.numpos=len(self.trlines)
        fd=open(basepath+"INRIAPerson/Test/neg.lst","r")
        self.trlines=self.trlines+fd.readlines()
        fd.close()
        
    def getDBname():
        return "INRIA POS"

    def getStorageDir(self):
        return self.local
        
    def getImage(self,i):
        item=self.trlines[i]
        impath=self.imagepath
        if i>=self.numpos:
            impath=self.imagepath2    
        return myimread((impath+item.split("/")[-1])[:-1])
    
    def getImageName(self,i):
        item=self.trlines[i]
        impath=self.imagepath
        if i>=self.numpos:
            impath=self.imagepath2 
        return (impath+item.split("/")[-1])[:-1]

    def getImageByName2(self,name):
        #item=self.trlines[i]
        impath=self.imagepath
        #if i>=self.numpos:
        #    impath=self.imagepath2 
        try:
            img=myimread(self.imagepath+name+".png")
        except:
            try:
                img=myimread(self.imagepath2+name+".png")
            except:
                img=myimread(self.imagepath2+name+".jpg")
        return img
    
    def getBBox(self,i):
        if i>=self.numpos:
            return []
        item=self.trlines[i]
        filename=self.annpath+item.split("/")[-1].split(".")[0]+".txt"
        return getbboxINRIA(filename)
    
    def getTotal(self):
        return len(self.trlines)

#class InriaTestFullParts(imageData):
#    """
#    INRIA database for positive examples
#    """
#    def __init__(self,numparts,part,basepath="/home/databases/",
#                        trainfile="INRIAPerson/Test/pos.lst",
#                        imagepath="INRIAPerson/Test/pos/",
#                        imagepath2="INRIAPerson/Test/neg/",
#                        local="INRIAPerson/",
#                        annpath="INRIAPerson/Test/annotations/"):
#        self.basepath=basepath
#        self.trainfile=basepath+trainfile
#        self.imagepath=basepath+imagepath
#        self.imagepath2=basepath+imagepath2
#        self.annpath=basepath+annpath
#        self.local=basepath+local
#        fd=open(self.trainfile,"r")
#        self.trlines=fd.readlines()
#        self.numpos=len(self.trlines)
#        self.numparts=numparts
#        self.part=part
#        fd=open(basepath+"INRIAPerson/Test/neg.lst","r")
#        self.trlines=self.trlines+fd.readlines()
#        fd.close()
#        
#    def getDBname():
#        return "INRIA POS"

#    def getStorageDir(self):
#        return self.local
#        
#    def getImage(self,i):
#        item=self.trlines[i*self.numparts+self.part]
#        impath=self.imagepath
#        if i*self.numparts+self.part>=self.numpos:
#            impath=self.imagepath2    
#        return myimread((impath+item.split("/")[-1])[:-1])
#    
#    def getImageName(self,i):
#        item=self.trlines[i*self.numparts+self.part]
#        impath=self.imagepath
#        if i*self.numparts+self.part>=self.numpos:
#            impath=self.imagepath2 
#        return (impath+item.split("/")[-1])[:-1]
#    
#    def getBBox(self,i):
#        if i*self.numparts+self.part>=self.numpos:
#            return []
#        item=self.trlines[i*self.numparts+self.part]
#        filename=self.annpath+item.split("/")[-1].split(".")[0]+".txt"
#        return getbboxINRIA(filename)
#    
#    def getTotal(self):
#        return len(self.trlines[self.part::self.numparts])

    
class VOC06Data(imageData):
    """
    VOC06 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,select="all",cl="person_train.txt",
                        basepath="meadi/DADES-2/",
                        trainfile="VOC2006/VOCdevkit/VOC2006/ImageSets/",
                        imagepath="VOC2006/VOCdevkit/VOC2006/PNGImages/",
                        annpath="VOC2006/VOCdevkit/VOC2006/Annotations/",
                        local="VOC2006/VOCdevkit/local/VOC2006/",
                        usetr=False,usedf=False,precompute=True):
        self.cl=cl
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile+cl
        self.imagepath=basepath+imagepath
        self.annpath=basepath+annpath
        self.prec=precompute
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        fd.close()
        if select=="all":#All images
            self.str=""
        if select=="pos":#Positives images
            self.str="1\n"
        if select=="neg":#Negatives images
            self.str="-1\n"
        self.selines=self.__selected()
        if self.prec:
            self.selbbox=self.__precompute()
        #sdf
    
    def __selected(self):
        lst=[]
        for id,it in enumerate(self.trlines):
            if self.str=="" or it.split(" ")[-1]==self.str:
                lst.append(it)
        return lst
    
    def __precompute(self):
        lst=[]
        tot=len(self.selines)
        cl=self.cl.split("_")[0]
        for id,it in enumerate(self.selines):
            print id,"/",tot
            filename=self.annpath+it.split(" ")[0]+".txt"
            #print filename
            #print getbboxVOC06(filename,cl,self.usetr,self.usedf)
            lst.append(getbboxVOC06(filename,cl,self.usetr,self.usedf))
        #raw_input()
        return lst

    def getDBname(self):
        return "VOC06"
        
    def getImage(self,i):
        item=self.selines[i]
        return myimread((self.imagepath+item.split(" ")[0])+".png")
    
    def getImageByName2(self,name):
        return myimread(self.imagepath+name+".png")

    def getImageByName(self,name):
        return myimread(name)
    
    def getImageName(self,i):
        item=self.selines[i]
        return (self.imagepath+item.split(" ")[0]+".png")

    def getImageRaw(self,i):
        item=self.selines[i]
        return im.open((self.imagepath+item.split(" ")[0])+".png")#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getStorageDir(self):
        return self.local
    
    def getBBox(self,i,cl=None,usetr=None,usedf=None):
        if usetr==None:
            usetr=self.usetr
        if usedf==None:
            usedf=self.usedf
        if cl==None:#use the right class
            cl=self.cl.split("_")[0]
        bb=[]
        if self.prec:
            bb=self.selbbox[i][:]
            #print self.selbbox
        else:
            item=self.selines[i]
            filename=self.annpath+item.split(" ")[0]+".txt"
            bb=getbboxVOC06(filename,cl,usetr,usedf)
        return bb

    def getBBoxByName(self,name,cl=None,usetr=None,usedf=None):
        if usetr==None:
            usetr=self.usetr
        if usedf==None:
            usedf=self.usedf
        if cl==None:#use the right class
            cl=self.cl.split("_")[0]
        filename=self.annpath+name+".txt"
        return getbboxVOC06(filename,cl,usetr,usedf)
            
    def getTotal(self):
        return len(self.selines)
    
#import Image as im

#VOCbase="/share/pascal2007/"

class VOC07Data(VOC06Data):
    """
    VOC07 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,select="all",cl="person_train.txt",
                basepath="media/DADES-2/",
                trainfile="VOC2007/VOCdevkit/VOC2007/ImageSets/Main/",
                imagepath="VOC2007/VOCdevkit/VOC2007/JPEGImages/",
                annpath="VOC2007/VOCdevkit/VOC2007/Annotations/",
                local="VOC2007/VOCdevkit/local/VOC2007/",
                usetr=False,usedf=False,mina=0):
        self.cl=cl
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile+cl
        self.imagepath=basepath+imagepath
        self.annpath=basepath+annpath
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        fd.close()
        if select=="all":#All images
            self.str=""
        if select=="pos":#Positives images
            self.str="1\n"
        if select=="neg":#Negatives images
            self.str="-1\n"
        self.selines=self.__selected()
        self.mina=mina
        
    def __selected(self):
        lst=[]
        for id,it in enumerate(self.trlines):
            if self.str=="" or it.split(" ")[-1]==self.str:
                lst.append(it)
        return lst

    def getDBname(self):
        return "VOC07"
    
    def getStorageDir(self):
        return self.local#"/media/DADES-2/VOC2007/VOCdevkit/local/VOC2007/"
        
    def getImage(self,i):
        item=self.selines[i]
        return myimread((self.imagepath+item.split(" ")[0])+".jpg")
    
    def getImageRaw(self,i):
        item=self.selines[i]
        return im.open((self.imagepath+item.split(" ")[0])+".jpg")#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageByName2(self,name):
        return myimread(self.imagepath+name+".jpg")

    def getImageName(self,i):
        item=self.selines[i]
        return (self.imagepath+item.split(" ")[0]+".jpg")
    
    def getTotal(self):
        return len(self.selines)
    
#    def getBBox(self,i,cl=None,usetr=None,usedf=None):
#        if usetr==None:
#            usetr=self.usetr
#        if usedf==None:
#            usedf=self.usedf
#        if cl==None:#use the right class
#            cl=self.cl.split("_")[0]
#        item=self.selines[i]
#        filename=self.annpath+item.split(" ")[0]+".xml"
#        return getbboxVOC07(filename,cl,usetr,usedf)

    def getBBox(self,i,cl=None,usetr=None,usedf=None):
        if usetr==None:
            usetr=self.usetr
        if usedf==None:
            usedf=self.usedf
        if cl==None:#use the right class
            cl=self.cl.split("_")[0]
        item=self.selines[i]
        filename=self.annpath+item.split(" ")[0]+".xml"
        bb=getbboxVOC07(filename,cl,usetr,usedf)
        auxb=[]
        for b in bb:
            a=abs(b[0]-b[2])*abs(b[1]-b[3])
            #print a
            if a>self.mina:
                #print "OK!"
                auxb.append(b)
        return auxb

class LFW(VOC06Data):
    """
    LFW
    """
    def __init__(self,select="all",cl="face_train.txt",
                basepath="media/DADES-2/",
                trainfile="lfw/lfw_ffd_ann.txt",
                imagepath="lfw/",
                annpath="lfw/",
                local="lfw/",
                usetr=False,usedf=False,mina=0,fold=0,totalfold=10,fake=False):
        self.fold=fold
        self.totalfold=totalfold
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile
        self.imagepath=basepath+imagepath
        self.annpath=basepath+annpath
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        fd.close()
        self.selines=self.trlines[6:]
        self.total=len(self.selines) #intial 5 lines of comments
        self.mina=mina
        self.fake=fake
        
    def __selected(self):
        lst=[]
        for id,it in enumerate(self.trlines):
            if self.str=="" or it.split(" ")[-1]==self.str:
                lst.append(it)
        return lst

    def getDBname(self):
        return "VOC07"
    
    def getStorageDir(self):
        return self.local#"/media/DADES-2/VOC2007/VOCdevkit/local/VOC2007/"
        
    def getImage(self,i):
        i=i+int(self.total/self.totalfold)*self.fold
        item=self.selines[i]
        return myimread((self.imagepath+item.split(" ")[0]))
    
    def getImageRaw(self,i):
        i=i+int(self.total/self.totalfold)*self.fold
        item=self.selines[i]
        return im.open((self.imagepath+item.split(" ")[0]))#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageByName2(self,name):
        return myimread(self.imagepath+name+".jpg")

    def getImageName(self,i):
        i=i+int(self.total/self.totalfold)*self.fold
        item=self.selines[i]
        return (self.imagepath+item.split(" ")[0])
    
    def getTotal(self):
        return int(self.total/self.totalfold)
    
#    def getBBox(self,i,cl=None,usetr=None,usedf=None):
#        if usetr==None:
#            usetr=self.usetr
#        if usedf==None:
#            usedf=self.usedf
#        if cl==None:#use the right class
#            cl=self.cl.split("_")[0]
#        item=self.selines[i]
#        filename=self.annpath+item.split(" ")[0]+".xml"
#        return getbboxVOC07(filename,cl,usetr,usedf)

    def getBBox(self,i,cl=None,usetr=None,usedf=None):
        if self.fake:
            #i=i+int(self.total/self.totalfold)*self.fold
            #item=self.selines[i]
            #im=myimread((self.imagepath+item.split(" ")[0]))
            im=LFW.getImage(self,i)
            dd=50
            bb=[[dd,dd,im.shape[0]-dd,im.shape[1]-dd,0,0]]
            return bb
        i=i+int(self.total/self.totalfold)*self.fold
        item=self.selines[i]
        aux=item.split()        
        cx=float(aux[1]);cy=float(aux[2]);w=float(aux[3]);h=float(aux[4])
        bb=[[cy,cx,cy+h,cx+w,0,0]]
        auxb=[]
        for b in bb:
            a=abs(float(b[0])-float(b[2]))*abs(float(b[1])-float(b[3]))
            #print a
            if a>self.mina:
                #print "OK!"
                auxb.append(b)
        return auxb

    def getPose(self,i):
        i=i+int(self.total/self.totalfold)*self.fold
        item=self.selines[i]
        aux=item.split()        
        return int(aux[5])


    def getFacial(self,i):
        i=i+int(self.total/self.totalfold)*self.fold
        item=self.selines[i]
        aux=item.split()        
        return (numpy.array(aux[7:7+int(aux[6])*2])).astype(numpy.float32)

class AFW(VOC06Data):
    """
    AFW
    """
    def __init__(self,select="all",cl="face_train.txt",
                basepath="media/DADES-2/",
                trainfile="afw/testimages/anno2.mat",
                imagepath="afw/testimages/",
                annpath="afw/testimages/",
                local="afw/",
                usetr=False,usedf=False,mina=0):
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile
        self.imagepath=basepath+imagepath
        self.annpath=basepath+annpath
        import util
        self.ann=util.loadmat(self.trainfile)["anno"]
        self.total=len(self.ann)
        self.mina=mina
        
    def getDBname(self):
        return "AFW"
    
    def getStorageDir(self):
        return self.local#"/media/DADES-2/VOC2007/VOCdevkit/local/VOC2007/"
        
    def getImage(self,i):
        item=self.ann[i][0][0]
        return myimread((self.imagepath+item))
    
    def getImageRaw(self,i):
        item=self.ann[i][0][0]
        return im.open((self.imagepath+item))#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageByName2(self,name):
        return myimread(self.imagepath+name+".jpg")

    def getImageName(self,i):
        item=self.ann[i][0][0]
        return (self.imagepath+item)
    
    def getTotal(self):
        return self.total
    
#    def getBBox(self,i,cl=None,usetr=None,usedf=None):
#        if usetr==None:
#            usetr=self.usetr
#        if usedf==None:
#            usedf=self.usedf
#        if cl==None:#use the right class
#            cl=self.cl.split("_")[0]
#        item=self.selines[i]
#        filename=self.annpath+item.split(" ")[0]+".xml"
#        return getbboxVOC07(filename,cl,usetr,usedf)

    def getBBox(self,i,cl=None,usetr=None,usedf=None):
        item=self.ann[i][1]
        bb=[]
        for l in range(item.shape[1]):
            it=item[0,l].flatten()
            bb.append([it[1],it[0],it[3],it[2],0,0])
        auxb=[]
        for b in bb:
            a=abs(float(b[0])-float(b[2]))*abs(float(b[1])-float(b[3]))
            #print a
            if a>self.mina:
                #print "OK!"
                auxb.append(b)
        return auxb

    def getPose(self,i):
        #i=i+int(self.total/self.totalfold)*self.fold
        #item=self.selines[i]
        #aux=item.split()        
        return self.ann[i][2][0][0][0]#int(aux[5])


    def getFacial(self,i):
        #i=i+int(self.total/self.totalfold)*self.fold
        #item=self.selines[i]
        #aux=item.split()        
        return self.ann[i][3][0].flatten()#(numpy.array(aux[7:7+int(aux[6])*2])).astype(numpy.float32)

class AFLW(VOC06Data):
    """
    AFLW
    """
    def __init__(self,select="all",cl="face_train.txt",
                basepath="media/DADES-2/",
                trainfile="aflw/data/aflw.sqlite",
                imagepath="aflw/data/flickr/",
                annpath="aflw/",
                local="aflw/",
                usetr=False,usedf=False,mina=0,fold=0):
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile
        self.imagepath=basepath+imagepath
        self.annpath=basepath+annpath
        import sqlite3 as lite
        #import util
        #self.ann=util.loadmat(self.trainfile)["anno"]
        #cnt=0
        #for l in self.ann:
        #    cnt+=l[1].shape[1]
        #self.total=len(self.ann)#cnt #intial 5 lines of comments
        self.mina=mina
        con = lite.connect(self.trainfile)
        self.cur = con.cursor() 
        #self.cur.execute("SELECT face_id FROM Faces")
        self.cur.execute("SELECT file_id FROM Faces")
        self.items=numpy.unique(self.cur.fetchall())
        self.total=len(self.items)
        
    def getDBname(self):
        return "AFLW"
    
    def getStorageDir(self):
        return self.local#"/media/DADES-2/VOC2007/VOCdevkit/local/VOC2007/"
        
    def getImage(self,i):
        #item=self.ann[i][0][0]
        #'SELECT db_id,filepath,width,height FROM FaceImages WHERE file_id =
        #self.cur.execute("SELECT file_id FROM Faces Where face_id = %d",items[i])
        #fileid=self.cur.fetchall()
        self.cur.execute("SELECT filepath FROM FaceImages WHERE file_id = '%s'"%self.items[i][0])
        impath=self.cur.fetchall()
        return myimread((impath))
    
    def getImageRaw(self,i):
        item=self.ann[i][0][0]
        return im.open((self.imagepath+item))#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageByName2(self,name):
        return myimread(self.imagepath+name+".jpg")

    def getImageName(self,i):
        #self.cur.execute("SELECT file_id FROM Faces Where face_id = %d"%self.items[i])
        #fileid=self.cur.fetchall()
        self.cur.execute("SELECT filepath FROM FaceImages WHERE file_id = '%s'"%self.items[i][0])
        impath=self.imagepath+self.cur.fetchall()[0][0]
        return (impath)
    
    def getTotal(self):
        return self.total
    
#    def getBBox(self,i,cl=None,usetr=None,usedf=None):
#        if usetr==None:
#            usetr=self.usetr
#        if usedf==None:
#            usedf=self.usedf
#        if cl==None:#use the right class
#            cl=self.cl.split("_")[0]
#        item=self.selines[i]
#        filename=self.annpath+item.split(" ")[0]+".xml"
#        return getbboxVOC07(filename,cl,usetr,usedf)

    def getBBox(self,i,cl=None,usetr=None,usedf=None):
        #SELECT x,y,w,h,annot_type_id FROM FaceRect WHERE face_id = 
        self.cur.execute("SELECT face_id FROM Faces WHERE file_id = '%s'"%self.items[i][0])
        faceid=self.cur.fetchall()
        bb=[]
        for l in faceid:   
            self.cur.execute("SELECT x,y,w,h,annot_type_id FROM FaceRect WHERE face_id = %d"%l)
            bb+=self.cur.fetchall()
        #bb.append([it[1],it[0],it[3],it[2],0,0])
        #print bb
        #raw_input()
        auxb=[]
        for b in bb:
            a=abs(float(b[0])-float(b[2]))*abs(float(b[1])-float(b[3]))
            #print a
            if a>self.mina:
                #print "OK!"
                auxb.append([b[1],b[0],b[1]+b[3],b[0]+b[2],0,0])
        return auxb

    def getPose(self,i):
        self.cur.execute("SELECT face_id FROM Faces WHERE file_id = '%s'"%self.items[i][0])
        faceid=self.cur.fetchall()
        poses=[]
        for l in faceid:
            self.cur.execute("SELECT roll,pitch,yaw FROM FacePose WHERE face_id = '%s'"%l)
            poses+=self.cur.fetchall()
        return poses


    def getFacial(self,i):
        self.cur.execute("SELECT face_id FROM Faces WHERE file_id = '%s'"%self.items[i][0])
        faceid=self.cur.fetchall()
        facial=[]
        for l in faceid:
            self.cur.execute("SELECT descr,FeatureCoords.x,FeatureCoords.y FROM FeatureCoords,FeatureCoordTypes WHERE face_id = '%s'"%l)
            facial+=self.cur.fetchall()
        return facial

class MultiPIE(VOC06Data):
    """
    MultiPIE
    """
    def __init__(self,select="all",cl="face_train.txt",
                basepath="media/DADES-2/",
                imagepath="multiPIE/MultiPIE/Multi-Pie/Multi-Pie/data/",
                session="session01",
                subject="001",
                recording="01",
                camera="05_1",
                usetr=False,usedf=False,mina=0,fold=0,ext="png"):
        self.camera=camera
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath
        self.imagepath=basepath+imagepath+session+"/multiview/"+subject+"/"+recording+"/"+camera
        #self.annpath=basepath+annpath
        self.selines=glob.glob(self.imagepath+"/*"+ext)
        self.selines.sort()
        self.total=len(self.selines)
        
    def getDBname(self):
        return "MultiPIE"
    
    def getStorageDir(self):
        return self.local#"/media/DADES-2/VOC2007/VOCdevkit/local/VOC2007/"
        
    def getImage(self,i):
        return myimread(self.selines[i])
    
    def getImageRaw(self,i):
        item=self.ann[i][0][0]
        return im.open((self.imagepath+item))#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageByName2(self,name):
        return myimread(self.imagepath+name)

    def getImageName(self,i):
        return (self.selines[i])
    
    def getTotal(self):
        return self.total
    
    def getBBox(self,i,cl=None,usetr=None,usedf=None):
        return [[120,240,320,440,0,0]]

    def getPose(self,i):
        if self.camera=="11_0":
            pose=-90
        elif self.camera=="12_0":
            pose=-75
        elif self.camera=="09_0":
            pose=-60
        elif self.camera=="08_0":
            pose=-45
        elif self.camera=="13_0":
            pose=-30
        elif self.camera=="14_0":
            pose=-15
        elif self.camera=="05_1":
            pose=0
        if self.camera=="05_0":
            pose=15
        elif self.camera=="04_1":
            pose=30
        elif self.camera=="19_0":
            pose=45
        elif self.camera=="20_0":
            pose=-60
        elif self.camera=="01_0":
            pose=75
        elif self.camera=="24_0":
            pose=90
        return pose


    def getFacial(self,i):
        return []


class Buffy(VOC07Data):

    def getImageName(self,i):
        item=self.selines[i]
        return (self.imagepath+item[:-2]+".jpg")

    def getBBox(self,i,cl="person",usetr=None,usedf=None):
        if usetr==None:
            usetr=self.usetr
        if usedf==None:
            usedf=self.usedf
        item=self.selines[i]
        filename=self.annpath+item[:-2]+".xml"
        bb=getbboxVOC07(filename,cl,usetr,usedf)
        auxb=[]
        for b in bb:
            a=abs(b[0]-b[2])*abs(b[1]-b[3])
            #print a
            if a>self.mina:
                #print "OK!"
                b1=(b[0]*2,b[1]*2,b[2]*2,b[3]*2,b[4],b[5],b[6])
                auxb.append(b1)
        return auxb
    

class imageNet(VOC07Data):

    def getImageName(self,i):
        item=self.selines[i]
        return (self.imagepath+item[:-1]+".JPEG")

    def getBBox(self,i,cl="person",usetr=None,usedf=None):
        if usetr==None:
            usetr=self.usetr
        if usedf==None:
            usedf=self.usedf
        item=self.selines[i]
        filename=self.annpath+item[:-1]+".xml"
        bb=getbboxVOC07(filename,"n02835271",usetr,usedf)
        return bb


class VOC11Data(VOC07Data):
    """
    VOC07 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,select="all",cl="person_train.txt",
                basepath="media/DADES-2/",
                trainfile="VOC2011/VOCdevkit/VOC2011/ImageSets/Main/",
                imagepath="VOC2011/VOCdevkit/VOC2011/JPEGImages/",
                annpath="VOC2011/VOCdevkit/VOC2011/Annotations/",
                local="VOC2011/VOCdevkit/local/VOC2011/",
                usetr=False,usedf=False):
        VOC07Data.__init__(self,select=select,cl=cl,basepath=basepath,
                trainfile=trainfile,imagepath=imagepath,
                annpath=annpath,local=local,usetr=usetr,usedf=usedf)
#        self.cl=cl
#        self.usetr=usetr
#        self.usedf=usedf
#        self.local=basepath+local
#        self.trainfile=basepath+trainfile+cl
#        self.imagepath=basepath+imagepath
#        self.annpath=basepath+annpath
#        fd=open(self.trainfile,"r")
#        self.trlines=fd.readlines()
#        fd.close()
#        if select=="all":#All images
#            self.str=""
#        if select=="pos":#Positives images
#            self.str="1\n"
#        if select=="neg":#Negatives images
#            self.str="-1\n"
#        self.selines=self.__selected()

class CVC02(VOC06Data):
    """
    CVC02 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,select="pos",
                basepath="/media/ca0567b8-ee6d-4590-8462-0d093addb4cf/DATASET-CVC-02/",
                trainfile="CVC-02-Classification/train/%s/color/",ext=".png",mirror=False,margin=0.1):
        if select=="pos":
            self.imagepath=basepath+trainfile%("positive")
        elif select=="neg":
            self.imagepath=basepath+trainfile%("negative-frames")
        elif select=="none":
            self.imagepath=basepath+trainfile
        self.ext=ext
        #fd=open(self.trainfile,"r")
        import glob
        if select!="pos":
            self.selines=glob.glob(self.imagepath+"/*"+self.ext)
        else:
            if mirror:
                self.selines=glob.glob(self.imagepath+"/*"+self.ext)
            else:
                self.selines=glob.glob(self.imagepath+"/positive-??????"+self.ext)#do not use mirror
        self.selines.sort()
        self.margin=margin
        #sdf
        #self.trlines=fd.readlines()
        #fd.close()

    def getDBname(self):
        return "CVC02pos"
    
    def getStorageDir(self):
        return ""#"/media/DADES-2/VOC2007/VOCdevkit/local/VOC2007/"
        
    def getImage(self,i):
        item=self.selines[i]
        return myimread(item)
    
#    def getImageRaw(self,i):
#        item=self.selines[i]
#        return im.open((self.imagepath+item.split(" ")[0])+self.ext)
    
    def getImageByName(self,name):
        return myimread(name)
    
#    def getImageByName2(self,name):
#        return myimread(self.imagepath+name+self.ext)

    def getImageName(self,i):
        item=self.selines[i]
        return (item)
    
    def getTotal(self):
        return len(self.selines)
    
    def getBBox(self,i,cl=None,usetr=None,usedf=None):
        item=self.selines[i]
        img=myimread(item)
        w=img.shape[1]
        h=img.shape[0]
        return [[0+self.margin*h,0+self.margin*h,h-self.margin*h,w-self.margin*h,0,0]]


class CVC02test(VOC06Data):
    """
    CVC02 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,select="pos",
                basepath="/media/ca0567b8-ee6d-4590-8462-0d093addb4cf/DATASET-CVC-02/",
                images="CVC-02-Classification/test-perimage/color/",
                annotations="CVC-02-Classification/test-perimage/annotations/",
                ext=".png"):
        self.imagepath=basepath+images
        self.annotations=basepath+annotations
        self.ext=ext
        import glob
        self.selines=glob.glob(self.imagepath+"/*"+self.ext)
        self.selines.sort()
        self.select=select
        #self.trlines=fd.readlines()

    def getDBname(self):
        return "CVC02test"
    
    def getStorageDir(self):
        return ""#"/media/DADES-2/VOC2007/VOCdevkit/local/VOC2007/"
        
    def getImage(self,i):
        item=self.selines[i]
        return myimread(item)
    
#    def getImageRaw(self,i):
#        item=self.selines[i]
#        return im.open((self.imagepath+item.split(" ")[0])+self.ext)
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageByName2(self,name):
        return myimread(self.imagepath+name+self.ext)

    def getImageName(self,i):
        item=self.selines[i]
        return (item)
    
    def getTotal(self):
        return len(self.selines)
    
    def getBBox(self,i,cl=None,usetr=None,usedf=None):
        item=self.selines[i]
        fd=open(self.annotations+item.split("/")[-1].split(".")[0]+".txt")
        bb=fd.readlines()
        ll=[]
        for l in bb:
            prs=l.split(" ")
            #print "prs:",'PEDESTRIAN-OBLIGATORY' in prs[-1]
            if 'PEDESTRIAN-OBLIGATORY' in prs[-1]:
                difficult=0
            else:
                #print "Consider this difficult becasue ",prs[-1]
                difficult=1
            ll.append([int(prs[1])-int(prs[3])/2,int(prs[0])-int(prs[2])/2,int(prs[1])+int(prs[3])/2,int(prs[0])+int(prs[2])/2,0,difficult])    
        return ll

class DirImages(VOC06Data):
    """
    VOC07 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,select="all",cl="person_train.txt",
                basepath="media/DADES-2/",
                trainfile="VOCdevkit/VOC2007/ImageSets/Main/",
                imagepath="VOCdevkit/VOC2007/JPEGImages/",
                annpath="",#VOCbase+"VOCdevkit/VOC2007/Annotations/",
                local="/tmp/",#VOCbase+"VOCdevkit/local/VOC2007/",
                usetr=False,usedf=False,ext=".png"):
        import glob
        self.cl=cl
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile+cl
        self.imagepath=imagepath
        self.annpath=basepath+annpath
        self.ext=ext
        self.selines=glob.glob(self.imagepath+"/*"+ext)
        self.selines.sort()
        
##    def __selected(self):
##        lst=[]
##        for id,it in enumerate(self.trlines):
##            if self.str=="" or it.split(" ")[-1]==self.str:
##                lst.append(it)
##        return lst

    def getDBname(self):
        return "Images"
    
    def getStorageDir(self):
        return self.local
        
    def getImage(self,i):
        item=self.selines[i]
        return myimread(item)
    
    def getBBox(self,i):
        return[]
    
##    def getImageRaw(self,i):
##        item=self.selines[i]
##        return im.open((self.imagepath+item.split(" ")[0])+".jpg")#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageName(self,i):
        item=self.selines[i]
        return (item)
    
    def getTotal(self):
        return len(self.selines)
    
##    def getBBox(self,i,cl=None,usetr=None,usedf=None):
##        if usetr==None:
##            usetr=self.usetr
##        if usedf==None:
##            usedf=self.usedf
##        if cl==None:#use the right class
##            cl=self.cl.split("_")[0]
##        item=self.selines[i]
##        filename=self.annpath+item.split(" ")[0]+".xml"
##        return getbboxVOC07(filename,cl,usetr,usedf)

import glob
import util

class CaltechData(imageData):
    """
    Caltech instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,select="all",cl="airplanes",num=10,nset=0,
                        basepath="/home/databases/101_ObjectCategories/",
                        #trainfile="VOC2006/VOCdevkit/VOC2006/ImageSets/",
                        #imagepath="VOC2006/VOCdevkit/VOC2006/PNGImages/",
                        #annpath="VOC2006/VOCdevkit/VOC2006/Annotations/",
                        local="local/"):
        self.cl=cl
        self.num=num
        self.path=basepath
        self.local=basepath+local
        self.select=select
        #import glob
        self.classes=glob.glob(basepath+"*")
        self.clidx=self.classes[0]
        self.clpos=0
        lf={}
        self.lts=[]
        if nset==-1:#random
            lf=glob.glob(basepath+cl+"/*.jpg")
            numpy.random.shuffle(lf)
        else:
            for acl in self.classes:
                acl1=acl.split("/")[-1]
                try:
                    
                    print "Loading class ",acl," set ",nset
                    lf[acl1]=util.load(basepath+acl1+"/sample%d"%nset)            
                except:
                    print "Loading failed",acl
                    lf[acl1]=glob.glob(basepath+acl1+"/*.jpg")
                    numpy.random.shuffle(lf[acl1])
                    util.save(basepath+acl1+"/sample%d"%nset,lf[acl1],prt=0)            
                    print "Shuffling and saving class ",acl," set ",nset
                self.lts=self.lts+lf[acl1][self.num:self.num+5]
        self.lf=lf
    
    def getDBname(self):
        return "Caltech101"
        
    def getImage(self,i):
        if self.select=="trpos":
        #item=self.selines[i]
            #print self.cl;raw_input()
            return myimread(self.lf[self.cl][i])
        if self.select=="trneg":
            acl=self.cl
            while (acl==self.cl):
                val=numpy.random.random_integers(len(self.classes))-1
                acl=self.classes[val].split("/")[-1]
            #alf=glob.glob(self.path+acl+"/*.jpg")
            #numpy.random.shuffle(alf)
            #print alf[i]
            return myimread(self.lf[acl][i%self.num])
        if self.select=="tspos":
        #item=self.selines[i]
            return myimread(self.lf[self.cl][i+self.num])
        if self.select=="tsall":
        #item=self.selines[i]
            #acl=self.cl
            #while (acl==self.cl):
            #    acl=self.classes[numpy.random.random_integers(len(self.classes))].split("/")       
            return myimread(self.lts[i])
    
    def getImageByName2(self,name):
        return myimread(self.imagepath+name+".png")

    def getImageByName(self,name):
        #if sel.select=="tsall":
        #    ll=self.lts[i].split("/")
        #    nstr="/".join(ll[:-1])+"/"+ll[-1].split("_")[-1]
        return myimread(name)
    
    def getImageName(self,i):
        if self.select=="trpos":
        #item=self.selines[i]
            #print self.cl;raw_input()
            return self.lf[self.cl][i]
        if self.select=="trneg":
            acl=self.cl
            while (acl==self.cl):
                acl=self.classes[numpy.random.random_integers(len(self.classes))-1].split("/")[-1]
            #alf=glob.glob(self.path+acl+"/*.jpg")
            #numpy.random.shuffle(alf)
            #print alf[i]
            return self.lf[acl][i%self.num]
        if self.select=="tspos":
            return self.lf[self.cl][i+self.num]
        if self.select=="tsall":
            #ll=self.lts[i].split("/")
            #nstr="/".join(ll[:-1])+"/"+ll[-2]+"_"+ll[-1]
            return self.lts[i]

    def getImageRaw(self,i):
        item=self.selines[i]
        return im.open((self.imagepath+item.split(" ")[0])+".png")#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getStorageDir(self):
        return self.local
    
    def getBBox(self,i,cl=None,usetr=None,usedf=None):
        if self.select=="tsall":
            if self.lts[i].split("/")[-2]!=self.cl:
                return []
            else:
                aux=myimread(self.lts[i])
                dy=aux.shape[0]/10
                dx=aux.shape[1]/10
                bb=[[0,0,aux.shape[0],aux.shape[1]]]#ymin,xmin,ymax,xmax
                return bb
        aux=myimread(self.lf[self.cl][i])
        dy=aux.shape[0]/10
        dx=aux.shape[1]/10
        #bb=[[0,0,aux.shape[0],aux.shape[1]]]#ymin,xmin,ymax,xmax
        bb=[[0+dy,0+dx,aux.shape[0]-dy,aux.shape[1]-dx]]#ymin,xmin,ymax,xmax
        return bb

    def getBBoxByName(self,name):#,cl=None,usetr=None,usedf=None):
        if self.select=="tsall":
            if name.split("/")[-2]!=self.cl:
                return []
            else:
                aux=myimread(self.lts[i])
                dy=aux.shape[0]/10
                dx=aux.shape[1]/10
                bb=[[0,0,aux.shape[0],aux.shape[1]]]#ymin,xmin,ymax,xmax
                return bb
        aux=myimread(name)
        dy=aux.shape[0]/10
        dx=aux.shape[1]/10
        return [[0+dy,0+dx,aux.shape[0]-dy,aux.shape[1]-dx]]
        #[[0,0,aux.shape[0],aux.shape[1]]]
            
    def getTotal(self):
        if self.select=="tspos":
            return len(self.lf[self.cl])-self.num
        if self.select=="tsall":
            return len(self.lts)
        return self.num

class WebCam(VOC06Data):
    """
    VOC07 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,select="all",cl="person_train.txt",
                basepath="media/DADES-2/",
                trainfile="VOCdevkit/VOC2007/ImageSets/Main/",
                imagepath="VOCdevkit/VOC2007/JPEGImages/",
                annpath="",#VOCbase+"VOCdevkit/VOC2007/Annotations/",
                local="/tmp/",#VOCbase+"VOCdevkit/local/VOC2007/",
                usetr=False,usedf=False,ext=".png",cv=None,capture=None):
        import glob
        self.cl=cl
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile+cl
        self.imagepath=imagepath
        self.annpath=basepath+annpath
        self.ext=ext
        self.selines=glob.glob(self.imagepath+"/*"+ext)
        self.c=0
        self.cv=cv
        self.capture=capture
        
    def getDBname(self):
        return "WebCam"
    
    def getStorageDir(self):
        return self.local
        
    def getImage(self,i):
        #item=self.selines[i]
        frame = self.cv.QueryFrame (self.capture)
        #self.cv.cvmat(frame)
        #print "after"
        img = numpy.asarray(self.cv.GetMat(frame))
        #img=img.astype(numpy.float)
        #img=opencv.cvIplImageAsNDarray(frame)
        self.c+=1
        return img
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageName(self,i):
        item="WebCam.%d"%self.c#self.selines[i]
        return (item)
    
    def getTotal(self):
        return 100000

#import camera    

class Player(VOC06Data):
    """
    VOC07 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,select="all",cl="person_train.txt",
                basepath="media/DADES-2/",
                trainfile="VOCdevkit/VOC2007/ImageSets/Main/",
                imagepath="VOCdevkit/VOC2007/JPEGImages/",
                annpath="",#VOCbase+"VOCdevkit/VOC2007/Annotations/",
                local="/tmp/",#VOCbase+"VOCdevkit/local/VOC2007/",
                usetr=False,usedf=False,ext=".png",cv=None,capture=None):
        import glob
        self.cl=cl
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile+cl
        self.imagepath=imagepath
        self.annpath=basepath+annpath
        self.ext=ext
        #self.selines=sort(glob.glob(self.imagepath+"/*"+ext))
        self.c=0
        self.client = camera.client_create(None,"158.109.9.201", 6665)
        #self.client = camera.client_create(None,"158.109.8.86", 6665)
        #self.client = camera.client_create(None,"158.109.9.212", 6665)
        camera.client_connect(self.client)
        self.cam = camera.camera_create(self.client, 0)
        #self.cam = camera.camera_create(self.client, 1)
        camera.camera_subscribe(self.cam, 1)
        
    def getDBname(self):
        return "WebCam"
    
    def getStorageDir(self):
        return self.local
        
    def getImage(self,i):
        #item=self.selines[i]
        #frame = self.cv.cvQueryFrame (self.capture)
        #img=self.cv.cvIplImageAsNDarray(frame)
        #self.c+=1
        camera.client_read(self.client)
        camera.camera_decompress(self.cam)
        #camera_save(cam, 'foo.ppm')
        # TESTING: This doesn't work...
        #print 'Width: %d'  % cam.contents.width;
        #print 'Height: %d' % cam.contents.height;
        data=camera.string_at(self.cam.contents.image,self.cam.contents.image_count)
        image=numpy.frombuffer(data,dtype=numpy.ubyte)
        image=image.reshape((self.cam.contents.height,self.cam.contents.width,3))
    #image1=image.copy()
        #pylab.clf()
        #pylab.ioff()
        #pylab.imshow(image)
        #pylab.show()
        #pylab.draw()
        #pylab.imshow(image)
        return image
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageName(self,i):
        item="WebCam.%d"%self.c#self.selines[i]
        return (item)
    
    def getTotal(self):
        return 100000
    

class ImgFile(VOC06Data):
    """
    Read images and BB from a pascal format detection file
    """
    def __init__(self,trainfile,imgpath="",local="/tmp/",sort=False,amin=400):
        self.trainfile=trainfile
        self.imgpath=imgpath
        import glob
        fd=open(self.trainfile,"r")
        trlines=fd.readlines()
        fd.close()
        images={}
        #limages=[]
        for l in trlines:
            line=l.split(" ")
            if len(line)>4:
                r=[int(line[2]),int(line[1]),int(line[4]),int(line[3]),0,0]
                a=(abs(r[3]-r[1])*abs(r[2]-r[0]))
                #print r
                #print "y",abs(r[4]-r[2]),"x",abs(r[3]-r[1]),"a",a
                if a>amin:#take only windows bigger than 400 pixles
                #if a>amin and r[2]>155:#take only windows bigger than 400 pixles
                    if images.has_key(line[0]):
                        images[line[0]].append(r)
                    else:
                        images[line[0]]=[r]
                #else:
                #    print "Examples Too small!!!"
        self.limages=([l for l in images.iterkeys()])#images.keys()#[l for l in images.iterkeys()]
        if sort:
            self.limages=sorted(self.limages)
        self.bbimages=images
        self.tot=len(self.limages)
        
    def getDBname(self):
        return "Images+BB"
    
    def getStorageDir(self):
        return self.local
        
    def getImage(self,i):
        return myimread(self.limages[i]+".png")
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageName(self,i):
        return self.imgpath+self.limages[i]+".png"
    
    def getBBox(self,i,cl=None,usetr=None,usedf=None):
        bb=self.bbimages[self.limages[i]]
        return bb

    def getTotal(self):
        return self.tot

#get image name and bbox
def extractInfo(trPosImages,maxnum=-1,usetr=True,usedf=False):
    bb=numpy.zeros((len(trPosImages)*20,4))#as maximum 5 persons per image in average
    name=[]
    cnt=0
    tot=0
    if maxnum==-1:
        tot=len(trPosImages)
    else:
        tot=min(maxnum,trPosImages)
    for idx in range(tot):
        #print trPosImages.getImageName(idx)
        #img=trPosImages.getImage(idx)
        rect=trPosImages[idx]["bbox"]#.getBBox(idx,usetr=usetr,usedf=usedf)
        for r in rect:
            bb[cnt,:]=r[:4]
            name.append(trPosImages[idx]["name"])#.getImageName(idx))
            cnt+=1
        #img=pylab.imread("circle.png")
        util.pdone(idx,tot)
    ratio=((bb[:,2])-(bb[:,0]))/((bb[:,3])-(bb[:,1]))
    area=((bb[:,2])-(bb[:,0]))*((bb[:,3])-(bb[:,1]))
    return name,bb[:cnt,:],ratio[:cnt],area[:cnt]


