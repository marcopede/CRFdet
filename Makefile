CC = gcc
CP = g++
#CC = icc

CFLAGS = -O3 -march=nocona -ffast-math -fomit-frame-pointer -fopenmp
#CFLAGS = -lm -msse2 -O2 -march=nocona -ffast-math -fomit-frame-pointer -fopenmp 

#OMPFLAGS = -fopenmp

#CC=icc
#CFLAGS = -xP -fast
#OMPFLAGS = -openmp

LIB_TARGETS = libresize.so libexcorr.so libhog.so libfastpegasos.so libcrf2.so
all:	$(LIB_TARGETS)

libcrf2.so: ./MRF2.1/myexample2.cpp Makefile
	$(CP) $(CFLAGS) -shared -Wl,-soname=libcrf2.so -DUSE_64_BIT_PTR_CAST -fPIC ./MRF2.1/myexample2.cpp ./MRF2.1/GCoptimization.cpp ./MRF2.1/maxflow.cpp ./MRF2.1/graph.cpp ./MRF2.1/LinkedBlockList.cpp ./MRF2.1/TRW-S.cpp ./MRF2.1/BP-S.cpp ./MRF2.1/ICM.cpp ./MRF2.1/MaxProdBP.cpp ./MRF2.1/mrf.cpp ./MRF2.1/regions-maxprod.cpp -o libcrf2.so

libexcorr.so: excorr.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libexcorr.so -fPIC excorr.c -o libexcorr.so #libmyrmf.so.1.0.1

libfastpegasos.so: fast_pegasos.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libfastpegasos.so -fPIC -lc fast_pegasos.c -o libfastpegasos.so

libresize.so:	resize.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libresize.so -fPIC resize.c -o libresize.so

libhog.so:	features2.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libhog.so -fPIC features2.c -o libhog.so

clean:
	rm -f *.o *.pyc $(EXE_TARGETS) $(LIB_TARGETS)


