//#############################################################################
//#
//# Fast_PD.h:
//#  Header file containing "CV_Fast_PD" class interface
//#  
//#############################################################################

#ifndef __FAST_PD_H__
#define __FAST_PD_H__

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include "graph.h"
#include "common.h"

//#define DIST(l0,l1)   (_dist[l1*_numlabels*4+l0*4])
#define WCOST(ed)       _wcosts[ed*4]
#define UPDATE_BALANCE_VAR0(y,d,h0,h1) { (y)+=(d); (h0)+=(d); (h1)-=d; }
#define NEW_LABEL(n) ((n)->parent && !((n)->is_sink))
#define REV(a) ((a)+1)
//#define dist(l1,l2,labx) &(_dist[l1*4*labx+l2*4])

//#############################################################################
//#
//# Classes & types
//#
//#############################################################################

//=============================================================================
// @class   CV_Fast_PD
// @author  Nikos Komodakis
//=============================================================================
//
class CV_Fast_PD
{

    public:

		//typedef Graph::Real Real;
        typedef float Real;

 //my code

inline void dist(int l1, int l2,int st_num_lab_x,Real *res)
{
    int x1=l1%st_num_lab_x;
    int y1=l1/st_num_lab_x;
    int x2=l2%st_num_lab_x;    
    int y2=l2/st_num_lab_x;
    res[0]=abs(y1-y2);    
    res[1]=abs(x1-x2);
    res[2]=(y1-y2)*(y1-y2);
    res[3]=(x1-x2)*(x1-x2);
    //forbid compression
    //res[0]=(y1-y2)<-1?1000:abs(y1-y2);    
    //res[1]=(x1-x2)<-1?1000:abs(x1-x2);
    //res[2]=(y1-y2)<-1?1000:(y1-y2)*(y1-y2);
    //res[3]=(x1-x2)<-1?1000:(x1-x2)*(x1-x2);
    res[0]=(y1-y2)>2?1000:abs(y1-y2);    
    res[1]=(x1-x2)>2?1000:abs(x1-x2);
    res[2]=(y1-y2)>2?1000:(y1-y2)*(y1-y2);
    res[3]=(x1-x2)>2?1000:(x1-x2)*(x1-x2);
}

inline Real dot(Real *a,Real *b)
{   
    Real r=0;
    for (int i=0;i<4;i++)
        r+=a[i]*b[i];
    return r;
}

inline void add(Real *a,Real *b, Real *c)
{
   for (int i=0;i<4;i++)
       c[i]=a[i]+b[i];
}

inline void addcum(Real *a,Real *b)
{
   for (int i=0;i<4;i++)
       a[i]+=b[i];
}

inline void neg(Real *a)
{
   for (int i=0;i<4;i++)
       a[i]=-a[i];
}

inline Real* getwcost(int ed)
{
    return &(_wcosts[ed*4]);
}
 
void scramble_labels(int *labels,int lenlabels)
{
   int r1,r2,temp;
   int num_times,cnt;

   num_times = lenlabels*2;
   //srand(clock());

   for ( cnt = 0; cnt < num_times; cnt++ )
   {
      r1 = rand()%lenlabels;  
      r2 = rand()%lenlabels;  

      temp             = labels[r1];
      labels[r1] = labels[r2];
      labels[r2] = temp;
   }
}


Real compute(int label,int l0,int l1,int i)
{
    Real aux1[4],aux2[4];
    dist(label,l1,_labx,aux1);
    dist(l0,label,_labx,aux2);
    addcum(aux1,aux2);
    dist(l0,l1,_labx,aux2);
    neg(aux2);
    addcum(aux1,aux2);
    Real delta = dot(getwcost(i),aux1);
    return delta;
}


		//
		// NOTE: "lcosts" is modified by member functions of this class
		//
        CV_Fast_PD( int numpoints, int numlabels, int labx, Real *lcosts,
		            int numpairs , int *pairs   , 
		            Real *distx   , int max_iters, 
		            Real *wcosts  )
        {
			int i;

			//printf( "Allocating memory...\n" );
			clock_t start = clock();

			// Init global vars and allocate memory
			//
        	_numpoints = numpoints;
            _numlabels = numlabels;
            _numpairs  = numpairs;
			_max_iters = max_iters;
			_dist      = distx;
            _pairs     = pairs;  
			_wcosts    = wcosts;
			_time      =-1; 
            _labx      = labx;
			_APF_change_time = -1;

			if ( _numlabels >= pow((float)256,(int)sizeof(Graph::Label)) ) 
			{
				printf( "\nChange Graph::Label type (it is too small to hold all labels)\n" );
				assert(0);
			}
				
			try
			{
                //allocate memory for distance
                //_dist = new Real[numlabels*numlabels*4];
				_children          = new Graph::node *[_numpoints];
				_source_nodes_tmp1 = new int[_numpoints]; 
				_source_nodes_tmp2 = new int[_numpoints]; 
			}
			catch(...)
			{
				printf( "\nError: cannot allocate memory...aborting\n" ); exit(0);
			}
            //fillup table
            //for (int l1 = 0; l1 < numlabels;l1++)
            //    for (int l2 = 0; l2 < numlabels;l2++)
            //        dist(l1,l2,_labx,&(_dist[l1*numlabels*4+l2*4]));
			for( i = 0; i < _numpoints; i++ )
			{
				_source_nodes_tmp1[i] = -2;
				_source_nodes_tmp2[i] = -2;
			}

			try
			{
				_all_nodes = new Graph::node[_numpoints*_numlabels];
				_all_arcs  = new Graph::arc[(_numpairs*_numlabels)<<1];
				_all_graphs = new Graph *[_numlabels];
			}
			catch(...)
			{
				printf( "\nError: cannot allocate memory...aborting\n" ); exit(0);
			}
			for( i = 0; i < _numlabels; i++ )
			{
				try
				{
					_all_graphs[i] = new Graph( &_all_nodes[_numpoints*i], &_all_arcs[(_numpairs*i)<<1], _numpoints, err_fun );
				}
				catch(...)
				{
					printf( "Error: cannot allocate memory...aborting\n" ); exit(0);
				}
                //printf( "Num labels %d\n",_numlabels );
				fillGraph( _all_graphs[i] );
				//if ( DIST(i,i) )
                /*if dist(i,i,)
				{
					printf( "Error: this version assumes that PAIRWISE_POTENTIAL(a,a) = 0 for any label a\n" );
					exit(0);
				}*/
			}
            //printf( "I am here...\n" );
			_active_list = -1;
			try
			{
				_pinfo = new Node_info[_numpoints];
			}
			catch(...)
			{
				printf( "\nError: cannot allocate memory...aborting\n" ); exit(0);
			}
			createNeighbors();

			_h = lcosts;
			try
			{
				_y = new Real[_numpairs*_numlabels];
			}
			catch(...)
			{
				printf( "\nError: cannot allocate memory...aborting\n" ); exit(0);
			}

			clock_t finish = clock();
			float t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
			//printf( "Done\n" );

			if ( t >= MAX((double)(_numpoints*MAX(_numlabels,5))/(double)(110000*16),1) )
			{
				printf( "========\n" );
				printf( "WARNING: algorithm may be very slow due to swapping from disk\n" ); 
				printf( "         (swapping will dominate the running time)\n" ); 
				printf( "========\n" );
			}
        }

		~CV_Fast_PD( void )
		{
			delete [] _all_nodes;
			delete [] _all_arcs ;

			Graph::Label i;
			for( i = 0; i < _numlabels; i++ )
				delete _all_graphs[i];
			delete [] _all_graphs;

			delete [] _einfo;
			delete [] _y;

			delete [] _pairs_arr;
			delete [] _pair_info;

			delete [] _pinfo;

			delete [] _source_nodes_tmp1;
			delete [] _source_nodes_tmp2;
			delete [] _children;
		}

		void init_duals_primals( void )
		{
			//printf( "Initializing..." );
            //random labels
            //srand(3);
            //int rlabels[_numlabels];
            //for (int l=0;l<_numlabels;l++)
            //    rlabels[l]=l;
            //scramble_labels(rlabels,_numlabels);
			// Set initial values for primal and dual variables
			int i;
			for( i = 0; i < _numpoints; i++ )
			{
				_pinfo[i].label =  0;
                //_pinfo[i].label =  rlabels[i];
                //_pinfo[i].label =  rand()%_numlabels;    
				_pinfo[i].time  = -1;
				_pinfo[i].next  = -1;
				_pinfo[i].prev  = -2;
			}

			memset( _y, 0, _numpairs*_numlabels*sizeof(Real) );
			for( i = 0; i < _numpairs; i++ )
			{
				int id0 = _einfo[i  ].tail;
				int id1 = _einfo[i  ].head;
				int l0  = _pinfo[id0].label;
				int l1  = _pinfo[id1].label;

				//Real d  = _wcosts[i]*DIST(l0,l1) - (_y[l0*_numpairs+i]-_y[l1*_numpairs+i]);
                //Real d  = WCOST(i)*DIST(l0,l1) - (_y[l0*_numpairs+i]-_y[l1*_numpairs+i]);
                Real aux[4];
                dist(l0,l1,_labx,aux);
                Real d  = dot(getwcost(i),aux) - (_y[l0*_numpairs+i]-_y[l1*_numpairs+i]);
				if ( l0 == l1 )
					assert( d == 0 );
				else
					UPDATE_BALANCE_VAR0( _y[l0*_numpairs+i], d, _h[l0*_numpoints+id0], _h[l0*_numpoints+id1] );

				_einfo[i].balance = -_y[l1*_numpairs+i];
			}

			// Get initial primal function
			//
			_APF = 0;
			for( i = 0; i < _numpoints; i++ )
			{
				_pinfo[i].height = _h[_pinfo[i].label*_numpoints+i];
				_APF += _pinfo[i].height;
			}

			//printf( "Done\n" );
		}

		void inner_iteration( Graph::Label label )
		{
			int i;

			Graph       *_graph =  _all_graphs[label];
			Graph::node *_nodes = &_all_nodes [_numpoints*label];
			Graph::arc  *_arcs  = &_all_arcs  [(_numpairs*label)<<1];
			Real        *_cur_y = &_y         [_numpairs*label];

			_time++;
			_graph->flow = 0;

			if ( _APF_change_time < _time - _numlabels )
				return;

			// Update balance and height variables
			//
			Arc_info  *einfo = _einfo;
			Graph::arc *arcs = _arcs;
			Real *cur_y = &_y[_numpairs*label];
			Real *cur_h = &_h[_numpoints*label];
			for( i = 0; i < _numpairs; i++, einfo++, arcs+=2, cur_y++ )
			{
				int l0,l1;
				if ( (l1=_pinfo[einfo->head].label) != label && (l0=_pinfo[einfo->tail].label) != label ) 
				{
					//Real delta  = _wcosts[i]*(DIST(label,l1)+DIST(l0,label)-DIST(l0,l1));
                    //Real delta  = WCOST(i)*(DIST(label,l1)+DIST(l0,label)-DIST(l0,l1));
                    Real delta = compute(label,l0,l1,i);//dot(getwcost(i)*aux1);

					//Real delta1 = _wcosts[i]*DIST(label,l1)-( (*cur_y)+einfo->balance);
                    //Real delta1 = WCOST(i)*DIST(label,l1)-( (*cur_y)+einfo->balance);
                    Real aux[4];
                    dist(label,l1,_labx,aux);
                    Real delta1 = dot(getwcost(i),aux)-( (*cur_y)+einfo->balance);
					Real delta2;
					if ( delta1 < 0 || (delta2=delta-delta1) < 0 ) 
					{
						UPDATE_BALANCE_VAR0( *cur_y, delta1, cur_h[einfo->tail], cur_h[einfo->head] )
						arcs->cap = arcs->r_cap = 0;
						if ( delta < 0 ) // This may happen only for non-metric distances
						{
							delta = 0;
							_nodes[einfo->head].conflict_time = _time;
							//_pair_info[i].conflict_time = _time;
						}
						REV(arcs)->r_cap = delta;
					}
					else
					{
						arcs->cap = arcs->r_cap = delta1;
						REV(arcs)->r_cap = delta2;
					}
				}
				else
				{
					arcs->cap = arcs->r_cap = 0;
					REV(arcs)->r_cap = 0;
				}
			}

			Real total_cap = 0;
			Node_info *pinfo = _pinfo;
			Graph::node *nodes = _nodes;
			for( i = 0; i < _numpoints; i++, pinfo++, nodes++, cur_h++ )
			{
				Real delta = pinfo->height - (*cur_h);
				nodes->tr_cap = delta;
				if (delta > 0) total_cap += delta;
			}
			
			// Run max-flow and update the primal variables
			//
			Graph::flowtype max_flow = _graph -> run_maxflow(1);
			_APF -= (total_cap - max_flow);
			if ( total_cap > max_flow )
				_APF_change_time = _time;

			cur_y = &_y[_numpairs*label];
			einfo =  _einfo;
			arcs  =  _arcs;
			for( i = 0; i < _numpairs; i++, einfo++, arcs+=2, cur_y++ )
				if ( _pinfo[einfo->head].label != label && _pinfo[einfo->tail].label != label )
				{
					if ( NEW_LABEL(&_nodes[einfo->head]) )
						einfo->balance = -(*cur_y + arcs->cap - arcs->r_cap);
				}
				else if ( _pinfo[einfo->head].label != label )
				{
					if ( NEW_LABEL(&_nodes[einfo->head]) )
						einfo->balance = -(*cur_y);
				}

			cur_h = &_h[_numpoints*label];
			pinfo =  _pinfo;
			nodes =  _nodes;
			for( i = 0; i < _numpoints; i++, pinfo++, nodes++, cur_h++ )
			{
				if ( pinfo->label != label )
				{
					if ( NEW_LABEL(nodes) )
					{
						// If necessary, repair "loads" in case of non-metric
						//
						if ( nodes->conflict_time > pinfo->time )
						{
							int k;
							for( k = 0; k < pinfo->num_pairs; k++)
							{
								int pid = pinfo->pairs[k];
								if ( pid <= 0 )
								{
									Pair_info *pair = &_pair_info[-pid];
									if ( !(_nodes[pair->i0].parent) || _nodes[pair->i0].is_sink)
									{
										Graph::Label l0 = _pinfo[pair->i0].label;
										Graph::Label l1 =  pinfo->label;
										//Real delta = _wcosts[-pid]*(DIST(l0,label)+DIST(label,l1)-DIST(l0,l1));
                                        //Real delta = WCOST(-pid)*(DIST(l0,label)+DIST(label,l1)-DIST(l0,l1));
                                        Real delta = compute(label,l0,l1,-pid);
                                        
										//assert( l0 != label ); assert( delta<0 );
										if ( delta < 0 )
										{
											_cur_y[-pid]  -= delta;
											_einfo[-pid].balance = -_cur_y[-pid];
											pinfo->height += delta; 
											_APF          += delta;
											_nodes[pair->i0].tr_cap += delta;
										}
									}
								}
							}
						}

						pinfo->label = label;
						pinfo->height -= nodes->tr_cap;
						nodes->tr_cap = 0;
						pinfo->time = _time;
					}
				}
				*cur_h = pinfo->height;
			}
		}

		void inner_iteration_adapted( Graph::Label label )
		{
			if ( _iter > 1 )
				return track_source_linked_nodes( label );

			int i;
			Graph       *_graph =  _all_graphs[label];
			Graph::node *_nodes = &_all_nodes [_numpoints*label];
			Graph::arc  *_arcs  = &_all_arcs  [(_numpairs*label)<<1];
			Real        *_cur_y = &_y         [_numpairs*label];
			Real        *_cur_h = &_h         [_numpoints*label];

			_time++;
			_graph->flow = 0;

			if ( _APF_change_time < _time - _numlabels )
				return;

			// Update dual vars (i.e. balance and height variables)
			//
			int dt = _time - _numlabels;
			for( i = 0; i < _numpairs; i++ )
			{
				int i0 = _pairs[ i<<1   ];
				int i1 = _pairs[(i<<1)+1];
				if ( _pinfo[i0].time >= dt || _pinfo[i1].time >= dt )
				{
					Graph::arc *arc0 = &_arcs[i<<1];

					if ( _cur_h[i0] != _pinfo[i0].height )
					{
						Real h = _cur_h[i0] - _nodes[i0].tr_cap;
						_nodes[i0].tr_cap = _pinfo[i0].height - h;
						_cur_h[i0] = _pinfo[i0].height;
					}

					if ( _cur_h[i1] != _pinfo[i1].height )
					{
						Real h = _cur_h[i1] - _nodes[i1].tr_cap;
						_nodes[i1].tr_cap = _pinfo[i1].height - h;
						_cur_h[i1] = _pinfo[i1].height;
					}

					int l0,l1;
					if ( (l0=_pinfo[i0].label) != label && (l1=_pinfo[i1].label) != label )
					{
						Graph::arc *arc1 = &_all_arcs[(_numpairs*l1+i)<<1];
						Real y_pq =   _cur_y[i] + arc0->cap - arc0->r_cap ;
						Real y_qp = -(_y[_numpairs*l1+i] + arc1->cap - arc1->r_cap);
						//Real delta  = _wcosts[i]*(DIST(label,l1)+DIST(l0,label)-DIST(l0,l1));
                        //Real delta  = WCOST(i)*(DIST(label,l1)+DIST(l0,label)-DIST(l0,l1));
                        Real delta = compute(label,l0,l1,i);
    
						//Real delta1 = _wcosts[i]*DIST(label,l1)-(y_pq+y_qp);
                        //Real delta1 = WCOST(i)*DIST(label,l1)-(y_pq+y_qp);
                        Real aux[4];
                        dist(label,l1,_labx,aux);
                        Real delta1 = dot(getwcost(i),aux)-(y_pq+y_qp);
						Real delta2;
						if ( delta1 < 0 || (delta2=delta-delta1) < 0 ) // is change necessary?
						{
							_cur_y[i] = y_pq+delta1;
							arc0->r_cap = arc0->cap = 0;
							if ( delta < 0 ) // This may happen only for non-metric distances
							{
								delta = 0;
								_nodes[i1].conflict_time = _time;
							}
							REV(arc0)->r_cap = delta;

							_nodes[i0].tr_cap -= delta1;
							_nodes[i1].tr_cap += delta1;
						}
						else
						{
							_cur_y[i] = y_pq;
							arc0->r_cap = arc0->cap = delta1;
							REV(arc0)->r_cap = delta2;
						}
					}
					else
					{
						_cur_y[i] += arc0->cap - arc0->r_cap;
						REV(arc0)->r_cap = arc0->r_cap = arc0->cap = 0;	
					}
				}
			}

			// Run max-flow and update the primal variables
			//
			assert( _iter <= 1 );
			//Graph::flowtype max_flow = _graph -> apply_maxflow(1);
            Graph::flowtype max_flow = _graph -> run_maxflow(1);

			double prev_APF = _APF;
			for( i = 0; i < _numpoints; i++ )
			{
				Node_info *pinfo = &_pinfo[i];
				if ( NEW_LABEL(&_nodes[i]) )
				{
					// If necessary, repair "loads" in case of non-metric
					//
					if ( _nodes[i].conflict_time > pinfo->time )
					{
						Real total_delta = 0;
						int k;
						for( k = 0; k < pinfo->num_pairs; k++)
						{
							int pid = pinfo->pairs[k];
							if ( pid <= 0 )
							{
								Pair_info *pair = &_pair_info[-pid];
								if ( !(_nodes[pair->i0].parent) || _nodes[pair->i0].is_sink)
								{
									Graph::Label l0 = _pinfo[pair->i0].label;
									Graph::Label l1 =  pinfo->label;
									//Real delta = _wcosts[-pid]*(DIST(l0,label)+DIST(label,l1)-DIST(l0,l1));
                                    //Real delta = WCOST(-pid)*(DIST(l0,label)+DIST(label,l1)-DIST(l0,l1));
                                    Real delta = compute(label,l0,l1,-pid);
									//printf( "pid = %d, delta = %d\n", -pid, delta );
									//assert( l0 != label ); assert( delta<0 );
									
									if ( delta < 0 )
									{
										_cur_y[-pid] -= delta;
										total_delta  += delta;
										_nodes[pair->i0].tr_cap += delta;
									}
								}
							}
						}
						if ( total_delta )
							_nodes[i].tr_cap -= total_delta;
					}

					pinfo->height -= _nodes[i].tr_cap; 
					_APF          -= _nodes[i].tr_cap;
					pinfo->time    = _time;
					pinfo->label   = label;

					if ( pinfo->prev == -2 ) // add to active list
					{
						pinfo->next = _active_list;
						pinfo->prev = -1;
						if (_active_list >= 0)
							_pinfo[_active_list].prev = i;
						_active_list = i;
					}
				}
			}
			if ( _APF < prev_APF )
				_APF_change_time = _time;
		}

		void track_source_linked_nodes( Graph::Label label )
		{
			int i;
			assert( _iter > 1 );

			//
			//
			Graph       *_graph =  _all_graphs[label];
			Graph::node *_nodes = &_all_nodes [_numpoints*label];
			Graph::arc  *_arcs  = &_all_arcs  [(_numpairs*label)<<1];
			Real        *_cur_y = &_y         [_numpairs*label];
			Real        *_cur_h = &_h         [_numpoints*label];

			//
			//
			_time++;
			_graph->flow = 0;

			if ( _APF_change_time < _time - _numlabels )
				return;

			int source_nodes_start1 = -1;
			int source_nodes_start2 = -1;

			// update active list of nodes
			// 
			int dt = _time - _numlabels;
			i = _active_list;
			while ( i >= 0 )
			{
				Node_info *n = &_pinfo[i];
				int i_next = n->next;

				if ( n->time >= dt )
				{
					if ( _cur_h[i] != n->height )
					{
						Real h = _cur_h[i] - _nodes[i].tr_cap;
						_nodes[i].tr_cap = n->height - h;
						_cur_h[i] = n->height;
					}

					if ( _nodes[i].tr_cap )
					{
						//assert(  _nodes[i].tr_cap < 0 );
						_nodes[i].parent  = TERMINAL;
						_nodes[i].is_sink = 1;
						_nodes[i].DIST = 1;
					}
					else _nodes[i].parent = NULL;
					//_nodes[i].TS = 0;
				}
				else // remove node from active list
				{
					int prev = n->prev;
					if ( prev >= 0 )
					{
						_pinfo[prev].next = n->next;
						if (n->next >= 0)
							_pinfo[n->next].prev = prev;
					}
					else 
					{
						_active_list = n->next;
						if ( _active_list >= 0 )
							_pinfo[_active_list].prev = -1;
					}
					n->prev = -2;
				}

				i = i_next;
			}

			// Update balance and height variables.
			// Also keep track and update source-linked-nodes.
			//
			i = _active_list;
			while ( i >= 0 )
			{
				Node_info *n = &_pinfo[i];
				int i_next = n->next;

				int k;
				Node_info *n0,*n1;
				for( k = 0; k < n->num_pairs; k++)
				{
					int i0,i1,ii;
					Pair_info *pair;
					int pid = n->pairs[k];
					if ( pid >= 0 )
					{
						pair = &_pair_info[pid];
						if ( pair->time == _time )
							continue;

						i0 = i; i1 = pair->i1;
						n0 = n; n1 = &_pinfo[i1];
						ii = i1;
					}
					else
					{
						pid = -pid;
						pair = &_pair_info[pid];
						if ( pair->time == _time )
							continue;

						i1 = i; i0 = pair->i0;
						n1 = n; n0 = &_pinfo[i0];
						ii = i0;
					}
					pair->time = _time;

					int l0,l1;
					Graph::arc *arc0 = &_arcs[pid<<1];
					if ( (l0=n0->label) != label && (l1=n1->label) != label )
					{
						Graph::arc *arc1 = &_all_arcs[(_numpairs*l1+pid)<<1];
						Real y_pq =   _cur_y[pid] + arc0->cap - arc0->r_cap ;
						Real y_qp = -(_y[_numpairs*l1+pid] + arc1->cap - arc1->r_cap);
						//Real delta  = _wcosts[pid]*(DIST(label,l1)+DIST(l0,label)-DIST(l0,l1));
                        //Real delta  = WCOST(pid)*(DIST(label,l1)+DIST(l0,label)-DIST(l0,l1));
                        Real delta = compute(label,l0,l1,pid);
						//Real delta1 = _wcosts[pid]*DIST(label,l1)-(y_pq+y_qp);
                        Real aux[4];
                        dist(label,l1,_labx,aux);
                        Real delta1 = dot(getwcost(pid),aux)-(y_pq+y_qp);
                         
						Real delta2;
						if ( delta1 < 0 || (delta2=delta-delta1) < 0 )
						{
							_cur_y[pid] = y_pq+delta1;
							arc0->r_cap = arc0->cap = 0;
							if ( delta < 0 ) // This may happen only for non-metric distances
							{
								delta = 0;
								_nodes[i1].conflict_time = _time;
							}
							REV(arc0)->r_cap = delta;

							_nodes[i0].tr_cap -= delta1;
							_nodes[i1].tr_cap += delta1;

							if ( _pinfo[ii].prev == -2 && _source_nodes_tmp2[ii] == -2 )
							{
								_source_nodes_tmp2[ii] = source_nodes_start2;
								source_nodes_start2 = ii;
							}
						}
						else
						{
							_cur_y[pid] = y_pq;
							arc0->r_cap = arc0->cap = delta1;
							REV(arc0)->r_cap = delta2;
						}
					}
					else
					{
						_cur_y[pid] += arc0->cap - arc0->r_cap;
						REV(arc0)->r_cap = arc0->r_cap = arc0->cap = 0;	
					}
				}

				Graph::node *nd = &_nodes[i];
				if ( nd->tr_cap > 0 )
				{
					nd -> is_sink = 0;
					nd -> parent = TERMINAL;
					nd -> DIST = 1;

					_graph->set_active(nd);
					
					_source_nodes_tmp1[i] = source_nodes_start1;
					source_nodes_start1 = i;
				}
				else if (nd->tr_cap < 0)
				{
					nd -> is_sink = 1;
					nd -> parent = TERMINAL;
					nd -> DIST = 1;
				}
				else nd -> parent = NULL;
				//n -> TS = 0;

				i = i_next;
			}

			for( i = source_nodes_start2; i >= 0; )
			{
				Graph::node *nd = &_nodes[i];
				if ( nd->tr_cap > 0 )
				{
					nd -> is_sink = 0;
					nd -> parent = TERMINAL;
					nd -> DIST = 1;

					_graph->set_active(nd);
					
					_source_nodes_tmp1[i] = source_nodes_start1;
					source_nodes_start1 = i;
				}
				else if (nd->tr_cap < 0)
				{
					nd -> is_sink = 1;
					nd -> parent = TERMINAL;
					nd -> DIST = 1;
				}
				else nd -> parent = NULL;
				//n -> TS = 0;
				
				int tmp = i;
				i = _source_nodes_tmp2[i];
				_source_nodes_tmp2[tmp] = -2;
			}

			// Run max-flow 
			//
			//Graph::flowtype max_flow = _graph -> apply_maxflow(0);
            Graph::flowtype max_flow = _graph -> run_maxflow(0);

			// Traverse source tree to update the primal variables 
			//
			double prev_APF = _APF;
			int num_children = 0;
			for( i = source_nodes_start1; i >= 0; )
			{
				Graph::node *n = &_nodes[i];
				if ( n->parent == TERMINAL )
				{
					Node_info *pinfo = &_pinfo[i];

					// If necessary, repair "loads" in case of non-metric
					//
					if ( n->conflict_time > pinfo->time )
					{
						Real total_delta = 0;
						int k;
						for( k = 0; k < pinfo->num_pairs; k++)
						{
							int pid = pinfo->pairs[k];
							if ( pid <= 0 )
							{
								Pair_info *pair = &_pair_info[-pid];
								if ( !(_nodes[pair->i0].parent) || _nodes[pair->i0].is_sink)
								{
									Graph::Label l0 = _pinfo[pair->i0].label;
									Graph::Label l1 =  pinfo->label;
									//Real delta = _wcosts[-pid]*(DIST(l0,label)+DIST(label,l1)-DIST(l0,l1));
                                    //Real delta = WCOST(-pid)*(DIST(l0,label)+DIST(label,l1)-DIST(l0,l1));
                                    Real delta = compute(label,l0,l1,-pid);
									//assert( l0 != label ); assert( delta<0 );
									
									if ( delta < 0 )
									{
										_cur_y[-pid] -= delta;
										total_delta  += delta;
										_nodes[pair->i0].tr_cap += delta;
									}
								}
							}
						}
						if ( total_delta )
							n->tr_cap -= total_delta;
					}

					pinfo->height   -= n->tr_cap; 
					_APF            -= n->tr_cap;
					pinfo->label     = label;
					pinfo->time      =_time;

					if ( pinfo->prev == -2 ) // add to active list
					{
						pinfo->next = _active_list;
						pinfo->prev = -1;
						if (_active_list >= 0)
							_pinfo[_active_list].prev = i;
						_active_list = i;
					}

					Graph::arc *a;
					for ( a=n->first; a; a=a->next )
					{
						Graph::node *ch = a->head;
						if ( ch->parent == a->sister )
							_children[num_children++] = ch;
					}
				}

				int tmp = i;
				i = _source_nodes_tmp1[i];
				_source_nodes_tmp1[tmp] = -2;
			}

			for( i = 0; i < num_children; i++ )
			{
				Graph::node *n = _children[i];
				//unsigned int id  = ((unsigned int)n - (unsigned int)_nodes) / sizeof(Graph::node); 
                unsigned long id  = ((unsigned long)n - (unsigned long)_nodes) / sizeof(Graph::node); 
				Node_info *pinfo = &_pinfo[id];

				// If necessary, repair "loads" in case of non-metric
				//
				if ( n->conflict_time > pinfo->time )
				{
					Real total_delta = 0;
					int k;
					for( k = 0; k < pinfo->num_pairs; k++)
					{
						int pid = pinfo->pairs[k];
						if ( pid <= 0 )
						{
							Pair_info *pair = &_pair_info[-pid];
							if ( !(_nodes[pair->i0].parent) || _nodes[pair->i0].is_sink)
							{
								Graph::Label l0 = _pinfo[pair->i0].label;
								Graph::Label l1 =  pinfo->label;
								//Real delta = _wcosts[-pid]*(DIST(l0,label)+DIST(label,l1)-DIST(l0,l1));
                                //Real delta = WCOST(-pid)*(DIST(l0,label)+DIST(label,l1)-DIST(l0,l1));
                                 Real delta = compute(label,l0,l1,-pid);
								//assert( l0 != label ); assert( delta<0 );

								if ( delta < 0 )
								{
									_cur_y[-pid] -= delta;
									total_delta  += delta;
									_nodes[pair->i0].tr_cap += delta;
								}
							}
						}
					}
					if ( total_delta )
						n->tr_cap -= total_delta;
				}

				pinfo->height   -= n->tr_cap; 
				_APF            -= n->tr_cap;
				pinfo->label     = label;
				pinfo->time      =_time;

				if ( pinfo->prev == -2 ) // add to active list
				{
					pinfo->next = _active_list;
					pinfo->prev = -1;
					if (_active_list >= 0)
						_pinfo[_active_list].prev = id;
					_active_list = id;
				}

				Graph::arc *a;
				for ( a=n->first; a; a=a->next )
				{
					Graph::node *ch = a->head;
					if ( ch->parent == a->sister )
						_children[num_children++] = ch;
				}
			}

			if ( _APF < prev_APF )
				_APF_change_time = _time;
		}

        void run( void )
        {
			double total_t = 0, total_augm = 0;
			init_duals_primals();
            //random ordering of the labels
            //srand(0);
            //int rlabels[_numlabels];
            //for (int l=0;l<_numlabels;l++)
            //    rlabels[l]=l;
			int iter = 0;
			while ( iter < _max_iters )
			{
                //scramble_labels(rlabels,_numlabels);
				double prev_APF = _APF;
				_iter = iter;

				//printf( "Iteration %d ", iter );
				clock_t start = clock();
				if ( !iter )
				{
					for( Graph::Label l = 0; l < _numlabels; l++ )
                    {
                        //printf("L=%d",rlabels[l]);
						inner_iteration( l );
                    }
				}
				else 
				{
					for( Graph::Label l = 0; l < _numlabels; l++ )
						inner_iteration_adapted( l );
				}
                //printf("R:(%d,%f) ",_pinfo[0].label,_h[_pinfo[0].label*_numpoints]);
				clock_t finish = clock();
				float t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
				total_t += t;
				//printf( "(%.3f secs)\n", t );

				if ( prev_APF <= _APF )
					break;

				iter++;
			}
            //printf("Energy Unary=%f\n",_APF);
			//printf( "Converged (total time = %f secs)\n", total_t );
        }

        Real score()
        {
            Real scr=0,scr1=0;
            for (int i=0;i<_numpoints;i++)//unary
            {
                //scr+=_h[i+_pinfo[i].label*_numpoints];
                scr+=_pinfo[i].height;
            //for (int i=0;i<_numpairs;i++)//pair
            //    scr1+=_y[i+_pinfo[i].label*_numpairs];
            //    printf("L%i: %f ",_pinfo[i].label,_h[_pinfo[i].label*_numpoints]);
            }
            //printf("Unary at label %d: %f\n",_pinfo[0].label+1,_h[(_pinfo[0].label+1)*_numpoints]);
            //printf("Scr:%f\n",scr);
            //printf("Balance:%f\n",scr1);
            //Real *aux=getwcost(0);
            //printf("WCOST %f %f %f %f",aux[0],aux[1],aux[2],aux[3]);
            return scr;
        }

		void fillGraph( Graph *_graph ) 
		{
			_graph->add_nodes();
			_graph->add_edges( _pairs, _numpairs );
		}

		void createNeighbors( void )
		{
			// Fill auxiliary structures related to neighbors
			//
			int i;
			try
			{
				_pairs_arr = new int[_numpairs*2];
			}
			catch(...)
			{
				printf( "\nError: cannot allocate memory...aborting\n" ); exit(0);
			}

			for( i = 0; i < _numpoints; i++ )
				_pinfo[i].num_pairs = 0;

			for( i = 0; i < _numpairs; i++ )
			{
				int i0 = _pairs[i<<1];
				int i1 = _pairs[(i<<1)+1];
				_pinfo[i0].num_pairs++; 
				_pinfo[i1].num_pairs++;
			}

			int offset = 0;
			for( i = 0; i < _numpoints; i++ )
			{
				_pinfo[i].pairs = &_pairs_arr[offset];  
				offset += _pinfo[i].num_pairs;
				_pinfo[i].num_pairs = 0;
			}

			try
			{
				_pair_info = new Pair_info[_numpairs];
				_einfo = new Arc_info[_numpairs];
			}
			catch(...)
			{
				printf( "\nError: cannot allocate memory...aborting\n" ); exit(0);
			}

			for( i = 0; i < _numpairs; i++ )
			{
				int i0 = _pairs[i<<1];
				int i1 = _pairs[(i<<1)+1];
				_pinfo[i0].pairs[_pinfo[i0].num_pairs++] =  i;
				_pinfo[i1].pairs[_pinfo[i1].num_pairs++] = -i;

				_einfo[i].tail = i0; 
				_einfo[i].head = i1;

				_pair_info[i].i0 = i0; 
				_pair_info[i].i1 = i1;
				_pair_info[i].time = -1;
			}
		}

		double MAX( double a, double b ) { return ( a >= b ? a : b ); }
		double MIN( double a, double b ) { return ( a <= b ? a : b ); }

		static void err_fun(char * msg)
		{
			printf("%s",msg);
		}

		struct Node_info
		{
			Graph::Label label; // current label
			Real height; // active height of node
			TIME time; // timestamp of change
 			int next;    
			int prev;
			int *pairs; // neighboring edges
			int num_pairs;
		};

        int           _numpoints;
        int           _numpairs;
        int           _numlabels;
        int           _labx;
		int           _max_iters;
        Real         *_h; // height variables
		Real         *_y; // balance variables
		Real         *_dist; // distance function for pairwise potential
        int          *_pairs;
		Graph::node  *_all_nodes; // Nodes and edges 
		Graph::arc   *_all_arcs;  // of max-flow graphs
		Graph       **_all_graphs;
		int          *_source_nodes_tmp1; // Auxiliary lists for keeping
		int          *_source_nodes_tmp2; // track of source-linked nodes
		Real         *_wcosts; // Weights of MRF pairwise potentials
		int           _iter;
		Node_info    *_pinfo; // info per MRF node

    private:

		// auxiliary data structures and variables
		//
		struct Pair_info
		{
			int i0, i1;
			TIME time;
		};

		struct Arc_info
		{
			int head, tail;
			Real balance;
		};

		Arc_info     *_einfo;
		double        _APF; // MRF energy
		int           _time;
		Pair_info    *_pair_info;
		int           _active_list;
		int           _APF_change_time;
		Graph::node **_children;
		int          *_pairs_arr;

        // Assignment or copying are not allowed
        //
        CV_Fast_PD( const CV_Fast_PD &other );
        CV_Fast_PD operator=( const CV_Fast_PD &other );
};


#endif /* __FAST_PD_H__ */

//#############################################################################
//#
//# EOF
//#
//#############################################################################

