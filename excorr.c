
#include<stdio.h>
#include<stdlib.h>
//#include<math.h>

#define ftype float
#define INFINITY 10000

static ftype k=1.0;//deformation coefficient

void setK(ftype pk)
{
    k=pk;
}

ftype getK()
{
    return k;
}

static compHOG=0; //number of HOG computed

void resetHOG()
{
    compHOG=0;
}

long getHOG()
{
    return compHOG;
}

//compute 3d correlation between an image feature img and a mask 
inline ftype corr3dpad(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,int posy,int posx,ftype *prec,int pady,int padx,int occl)
{
    int dimzfull=dimz;
    if (occl!=0)
        //with occl
        dimz=dimz-occl;
        //printf("Occl:%d Dimzfull:%d Dimz:%d",occl,dimzfull,dimz);
    if (prec!=NULL)//memoization of the already computed locations
    {
        if (posy>=-pady && posy<imgy+pady && posx>=-padx && posx<imgx+padx)
            if (prec[(posy+pady)*(imgx+2*padx)+(posx+padx)]>-INFINITY)
            {
                return prec[(posy+pady)*(imgx+2*padx)+(posx+padx)];
            }
    }    
    ftype sum=0.0;
    int x,y,z,posi;
    for (x=0;x<maskx;x++)
        for (y=0;y<masky;y++)
        {
            compHOG++;
            if (((x+posx)>=0 && (x+posx<imgx)) && ((y+posy)>=0 && (y+posy<imgy)))
            //inside the image
            {
                for (z=0;z<dimz;z++)
                {   
                    //printf("%d:%f\n",z,mask[z+x*dimzfull+y*dimzfull*maskx]);
                    posi=z+(x+posx)*dimz+(y+posy)*dimz*imgx;
                    sum=sum+img[posi]*mask[z+x*dimzfull+y*dimzfull*maskx];      
                    /*posi=z+(x+posx)*dimzfull+(y+posy)*dimzfull*imgx;
                    {
                        sum=sum+img[posi]*mask[z+x*dimzfull+y*dimzfull*maskx];      
                    }*/
                }
            }
            else
            //occlusion using dimz
            {
                for (z=dimz;z<dimzfull;z++)
                {
                    //printf("%d:%f\n",z,mask[z+x*dimzfull+y*dimzfull*maskx]);
                    //posi=z+(x+posx)*dimz+(y+posy)*dimz*imgx;
                    sum=sum+mask[z+x*dimzfull+y*dimzfull*maskx];      
                }
            }
        }
    if (prec!=NULL)//save computed values in the buffer
    {
        if (posy>=-pady && posy<imgy+pady && posx>=-padx && posx<imgx+padx)
        {
            prec[(posy+pady)*(imgx+2*padx)+(posx+padx)]=sum;
        }
    }
    return sum;
}


static ftype buffer[5*5*13];//for BOW part should be smaller than 50
static int bcode[50*50];//for BOW
static ftype localhist[2000];
//static int table[10000];//(9 max orient+ null)^4
//static variables for the moment
//int sizevoc=2;
//int numvoc=100;
//ftype *voc
//ftype *mhist

/*void filltable(int *tab)
{
    int c;
    for (c=0;c<625;c++)
        table[c]=tab[c];
}*/

#define NUMOR 5
static int sel[NUMOR]={0,2,4,5,7};//selected orientations
static int norm[4]={2,3,0,1};//selected orientations
static ftype pow5[]={1,5,5*5,5*5*5};
static ftype pow6[]={1,6,6*6,6*6*6};


//compute the score over the possible values of a defined neighborhood
inline ftype refineigh(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,int posy,int posx,int rady,int radx,int *posedy, int *posedx, ftype *prec,int occl)
{
    int iy,ix;  
    ftype val,maxval=-1000;
    for (iy=-rady;iy<=rady;iy++)
    {
        for (ix=-radx;ix<=radx;ix++)
        {
            val=corr3dpad(img,imgy,imgx,mask,masky,maskx,dimz,posy+iy,posx+ix,prec,0,0,occl);
            if (val>maxval)
            {
                maxval=val;
                *posedy=iy;
                *posedx=ix;
            }
        }
    }
    return maxval;
}

void scaneigh(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,int *posy,int *posx,ftype *val,int *posey,int *posex,int rady, int radx,int len,int occl)
{   
    //return;
    int i;
    for (i=0;i<len;i++)
    {
        //if (posy[i]==-1  && posx[i]==-1)
        //    val[i]=0.0;
        //else
        val[i]=refineigh(img,imgy,imgx,mask,masky,maskx,dimz,posy[i],posx[i],rady,radx,posey++,posex++,NULL,occl);       
    }
}

inline ftype refineighfull(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,ftype dy,ftype dx,int posy,int posx,int rady,int radx,ftype *scr,int *rdy,int *rdx,ftype *prec,int pady,int padx,int occl)
{
    int iy,ix;  
    ftype val,maxval=-1000;
    //printf("dy:%d,dx:%d",rady,radx);
    //printf("k=%f",k);
    for (iy=-rady;iy<=rady;iy++)
    {
        for (ix=-radx;ix<=radx;ix++)
        {
            val=corr3dpad(img,imgy,imgx,mask,masky,maskx,dimz,posy+iy,posx+ix,prec,pady,padx,occl)+k*k*dy*(iy*iy)+k*k*dx*(ix*ix);
            scr[(iy+rady)*(2*radx+1)+(ix+radx)]+=-val;
            if (val>maxval)
            {
                maxval=val;
                *rdy=iy;
                *rdx=ix;
            }
        }
    }
    return maxval;
}


