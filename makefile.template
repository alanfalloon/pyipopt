CC = gcc
CFLAGS = -O3 -fpic -shared
DFLAGS = -fpic -shared
LDFLAGS = -lipopt  -lm -lblas -llapack
PY_DIR = /usr/local/lib/python2.5/site-packages

# Change this to your ipopt include path that includes IpStdCInterface.h 
IPOPT_INCLUDE = /data/walter/opt_software/ipopt/include/coin
IPOPT_LIB =    /data/walter/opt_software/ipopt/lib


# Change this to your python dir which includes Python.h
# You might need to download the python source code or install python-dev to get
# this header file. Note that Pyipopt needs this as an extend python module. 
PYTHON_INCLUDE = /usr/include/python2.5

# Change this to your numpy include path which contains numpy/arrayobject.h
# If you don't want this and would like to use list. I have a nasty (and unmaintained version) in the package called pyipopt-list.c. You can compile it and use it without numpy. 
# However, numpy is strongly suggested. 

NUMPY_INCLUDE = /usr/lib/python2.5/site-packages/numpy/core/include

pyipopt: callback.c pyipopt.c
	$(CC) -o pyipopt.so -Wl,--rpath,$(IPOPT_LIB) -I$(PYTHON_INCLUDE) -I$(IPOPT_INCLUDE) -I$(NUMPY_INCLUDE) $(CFLAGS) -L$(IPOPT_LIB) $(LDFLAGS) pyipopt.c callback.c

debug: callback.c pyipopt.c
	$(CC) -g -o pyipopt.so -I$(PYTHON_INCLUDE) -I$(IPOPT_INCLUDE) -I$(NUMPY_INCLUDE) $(DFLAGS) $(LDFLAGS) pyipopt_debug.c callback.c

debug_install: debug
	cp ./pyipopt.so $(PY_DIR)

install: pyipopt
	cp ./pyipopt.so $(PY_DIR)
clean:
	rm pyipopt.so 
