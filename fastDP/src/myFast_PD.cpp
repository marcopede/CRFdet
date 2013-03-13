//---------------------------------------------------------------------------

#include <stdio.h>
#include <vector>
#include "Fast_PD.h"

typedef CV_Fast_PD::Real Real;
typedef Real dtype;

extern "C" {

dtype compute_graph(int num_parts_y,int num_parts_x,int num_pairs,int *pairs,int num_lab_y,int num_lab_x,Real *data,Real *dist,Real *wcosts,
int numhyp,Real* lscr,int *reslab,int aiter,int restart)
{
    double lowerBound;
    float t,tot_t;
    int iter;

    int seed = 1124285485;
    srand(seed);

    dtype scr;

    int st_parts_x=num_parts_x;
    int st_parts_y=num_parts_y;
    int st_num_lab_x=num_lab_x;
    int st_num_lab_y=num_lab_y;
    int st_numlab=num_lab_y*num_lab_x;//num_parts_x*num_parts_y;//num_labels;
    //copy data because it is internally modified
    std::vector<dtype> copy_data_vec(data, data + num_parts_x * num_parts_y * num_lab_x * num_lab_y);
    dtype* copy_data = copy_data_vec.data();
//    dtype* copy_data=(dtype*)malloc(num_parts_x*num_parts_y*num_lab_x*num_lab_y*sizeof(dtype));
//    for (int i=0;i<num_parts_x*num_parts_y*num_lab_x*num_lab_y;i++)
//        copy_data[i]=data[i];
    clock_t t0,t1;
    t0 = clock ();
    int l1,l2;
    CV_Fast_PD pd( num_parts_x*num_parts_y, num_lab_x*num_lab_y, num_lab_x, copy_data,
	               num_pairs, pairs, dist, aiter,
				   wcosts );
  	//CV_Fast_PD pd( _numpoints, _numlabels, _lcosts,
	//               _numpairs, _pairs, _dist, _max_iters,
	//			   _wcosts );
    //printf("Parts %d,%d Labels %d\n",num_parts_x,num_parts_y,num_lab_x*num_lab_y);
	pd.run();
    for( int i = 0; i < num_parts_x*num_parts_y; i++ )
    {
        //printf("Lab%d:%d ",i,pd._pinfo[i].label);
	 	reslab[i]= pd._pinfo[i].label;  
    }

    Real bestE=pd.score(); 
	/*for (iter=0; iter<numhyp; iter++) 
    {
        bestE=10000;
        for (int idrestart=restart; idrestart>=0;idrestart--)
        {
            if (restart==0)
		        mrf->optimize(aiter, t);
		    else
		    {
			    //mrf->initialize();
                ((Expansion*)mrf)->clearAnswer();
			    mrf->optimize(aiter,t);	
		    }
                //printf("After C\n");
      		E = mrf->totalEnergy();
            //printf("After(%d,%d)",num_lab_y,num_lab_x);
            if (restart>0 && E>bestE)
            {
                //printf("Higher energy it=%d rest=%d energy=%f\n",iter,idrestart,E);
                continue;
            }
            //printf("Lower energy it=%d rest=%d energy=%f\n",iter,idrestart,E);
            bestE=E;
       		//lowerBound = mrf->lowerBound();
       		tot_t = tot_t + t ;
            for ( int  i = 0; i < num_parts_y*num_parts_x; i++ )
            {
                reslab[iter*num_parts_y*num_parts_x+i] = mrf->getLabel(i);//gc->whatLabel(i);
            }
	    }        
	    *lscr=-bestE;
	    lscr++;
        //t0 = clock ();
        for ( int  i = 0; i < num_parts_y*num_parts_x; i++ )
        {
            int aux=reslab[iter*num_parts_y*num_parts_x+i];
			//reslab[iter*num_parts_y*num_parts_x+i] = aux;//gc->whatLabel(i);
            //data[i*st_numlab+aux]=1;//delete solution
            for (int rx=-1;rx<2;rx++)
            {
                for (int ry=-1;ry<2;ry++)  
                {                       
                    int pp=aux+rx+ry*st_num_lab_x;
                    pp=maxi(pp,0);
                    pp=mini(pp,st_numlab-1);
                    data[pp+i*st_numlab]=10;//delete solution
                }
            }
            //printf("%d ",reslab[i]);
        }
        //printf("Done\n",num_lab_y,num_lab_x);
        //t1 = clock ();   
        //printf("t0=%d t1=%d Diff %f \n",t0,t1,float(t1-t0)/CLOCKS_PER_SEC);
        //trees=((Expansion*)mrf)->getEnergies();
        //sol=((Expansion*)mrf)->getSolutions();        
    }
    delete mrf;*/
    t1 = clock ();
    //printf("t0=%d t1=%d Diff %f \n",t0,t1,float(t1-t0)/CLOCKS_PER_SEC);
    return -bestE;
}

} //end extern C


