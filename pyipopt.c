/* Copyright (c) 2008, Eric You Xu, Washington University
* All rights reserved.
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the Washington University nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS" AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE REGENTS AND CONTRIBUTORS BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "hook.h"

/* Object Section */
// sig of this is void foo(PyO*)
static void problem_dealloc(PyObject* self)
{
	problem* temp = (problem*)self;
	free(temp->data);
	return;
}

PyObject* solve (PyObject* self, PyObject* args);
PyObject* close_model (PyObject* self, PyObject* args);

static char PYIPOPT_SOLVE_DOC[] = "solve(x) -> (x, ml, mu, obj)\n \
        \n \
        Call Ipopt to solve problem created before and return  \n \
        a tuple that contains final solution x, upper and lower\n \
        bound for multiplier and final objective function obj. ";

static char PYIPOPT_CLOSE_DOC[] = "After all the solving, close the model\n";

static char PYIPOPT_ADD_STR_OPTION_DOC[] = "Set the String option for Ipopt. See the document for Ipopt for more information.\n";


PyObject *add_str_option(PyObject *self, PyObject *args)
{
  	problem* temp = (problem*)self; 	
  	IpoptProblem nlp = (IpoptProblem)(temp->nlp);
  	
  	char* param;
  	char* value;
  	
  	Bool ret;
  	
  	if (!PyArg_ParseTuple(args, "ss", &param, &value))
  	{
     	Py_INCREF (Py_False);
        return Py_False;
    }
  	ret = AddIpoptStrOption(nlp, (char*) param, value);
	if (ret) 
	{
		Py_INCREF(Py_True);
		return Py_True;
	}
	else 
	{
		Py_INCREF(Py_False);
		return Py_False;
	}
}


static char PYIPOPT_ADD_INT_OPTION_DOC[] = "Set the Int option for Ipopt. See the document for Ipopt for more information.\n";

PyObject *add_int_option(PyObject *self, PyObject *args)
{
  	problem* temp = (problem*)self; 	
  	IpoptProblem nlp = (IpoptProblem)(temp->nlp);
  	
  	char* param;
  	int value;
  	
  	Bool ret;
  	
  	if (!PyArg_ParseTuple(args, "si", &param, &value))
        {
  		Py_INCREF(Py_False);
                return Py_False;
        }

  	ret = AddIpoptIntOption(nlp, (char*) param, value);
	if (ret) 
	{
		Py_INCREF(Py_True);
		return Py_True;
	}
	else 
	{
		Py_INCREF(Py_False);
		return Py_False;
	}
}


static char PYIPOPT_ADD_NUM_OPTION_DOC[] = "Set the Number/double option for Ipopt. See the document for Ipopt for more information.\n";

PyObject *add_num_option(PyObject *self, PyObject *args)
{
  	problem* temp = (problem*)self; 	
  	IpoptProblem nlp = (IpoptProblem)(temp->nlp);
  	
  	char* param;
  	double value = 1.;
  	
  	Bool ret;

	if (!PyArg_ParseTuple(args, "sd:num_option", &param, &value))
	{
		return NULL;
	}

 	ret = AddIpoptNumOption(nlp, (char*) param, value);
	if (ret) 
	{
		Py_INCREF(Py_True);
		return Py_True;
	}
	else 
	{
		Py_INCREF(Py_False);
		return Py_False;
	}
}



PyMethodDef problem_methods[] = {
	{ "solve", 	solve, METH_VARARGS, PYIPOPT_SOLVE_DOC},
	{ "close",  close_model, METH_VARARGS, PYIPOPT_CLOSE_DOC}, 
	{ "int_option", add_int_option, METH_VARARGS, PYIPOPT_ADD_INT_OPTION_DOC},
	{ "str_option", add_str_option, METH_VARARGS, PYIPOPT_ADD_STR_OPTION_DOC},
	{ "num_option", add_num_option, METH_VARARGS, PYIPOPT_ADD_NUM_OPTION_DOC},
	{NULL, NULL},
};

PyObject *problem_getattr(PyObject* self, char* attrname)
{ 
	PyObject *result = NULL;
    result = Py_FindMethod(problem_methods, self, attrname);
    return result;
}

PyTypeObject IpoptProblemType = {
    PyObject_HEAD_INIT(&PyType_Type)
    0,                         /*ob_size*/
    "pyipopt.Problem",         /*tp_name*/
    sizeof(problem),    		/*tp_basicsize*/
    0,                         /*tp_itemsize*/
    problem_dealloc,           /*tp_dealloc*/
    0,                         /*tp_print*/
    problem_getattr,           /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,        /*tp_flags*/
    "The IPOPT problem object in python", /* tp_doc */
};

static char PYIPOPT_CREATE_DOC[] = "create(n, xl, xu, m, gl, gu, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g) -> Boolean\n \
        \n \
        Create a problem instance and return True if succeed  \n \
        \n \
        n is the number of variables, \n \
        xl is the lower bound of x as bounded constraints \n \
        xu is the upper bound of x as bounded constraints \n \
        	both xl, xu should be one dimension arrays with length n \n \
        \n \
        m is the number of constraints, \n \
        gl is the lower bound of constraints \n \
        gu is the upper bound of constraints \n \
        	both gl, gu should be one dimension arrays with length m \n \
        nnzj is the number of nonzeros in Jacobi matrix \n \
        nnzh is the number of non-zeros in Hessian matrix, you can set it to 0 \n \
        \n \
        eval_f is the call back function to calculate objective value, \n \
        	it takes one single argument x as input vector \n \
        eval_grad_f calculates gradient for objective function \n \
        eval_g calculates the constraint values and return an array \n \
        eval_jac_g calculates the Jacobi matrix. It takes two arguments, \n \
        	the first is the variable x and the second is a Boolean flag \n \
        	if the flag is true, it supposed to return a tuple (row, col) \n \
        		to indicate the sparse Jacobi matrix's structure. \n \
        	if the flag is false if returns the values of the Jacobi matrix \n \
        		with length nnzj \n \
        eval_h calculates the hessian matrix, it's optional. \n \
        	if omitted, please set nnzh to 0 and Ipopt will use approximated hessian \n \
        	which will make the convergence slower. ";
        	
static PyObject *create(PyObject *obj, PyObject *args)
{
	PyObject *f; 
	PyObject *gradf;
	PyObject *g;
	PyObject *jacg;
	PyObject *h = NULL;
	PyObject *applynew = NULL;
	
	DispatchData myowndata;
	
	// I have to create a new python object here, return this python object 
	
	int n;			// Number of var
	PyArrayObject *xL;
	PyArrayObject *xU;
	int m;			// Number of con
	PyArrayObject *gL;
	PyArrayObject *gU;
	
	int nele_jac;
	int nele_hess;
	
	double* xldata, *xudata;
	double* gldata, *gudata;
	
	double result;
	int i;
    
	// Init the myowndata field
	myowndata.eval_f_python = NULL;
	myowndata.eval_grad_f_python = NULL; 
	myowndata.eval_g_python = NULL;
	myowndata.eval_jac_g_python = NULL;
	myowndata.eval_h_python = NULL;
	myowndata.apply_new_python = NULL;
	myowndata.userdata = NULL;
    
	// "O!", &PyArray_Type &a_x 
	if (!PyArg_ParseTuple(args, "iO!O!iO!O!iiOOOO|OO", 
			      &n, &PyArray_Type, &xL, 
			      &PyArray_Type, &xU, 
			      &m, 
			      &PyArray_Type, &gL,
			      &PyArray_Type, &gU,
			      &nele_jac, &nele_hess,
			      &f, &gradf, &g, &jacg, 
			      &h, &applynew)) 
	{
		return NULL;
	}    
        
	if (!PyCallable_Check(f)     ||
	    !PyCallable_Check(gradf) || 
	    !PyCallable_Check(g)     ||
	    !PyCallable_Check(jacg))
	{
		PyErr_SetString(PyExc_TypeError, 
				"Need a callable object for function!");
		return NULL;
	}
	myowndata.eval_f_python      = f;
	myowndata.eval_grad_f_python = gradf;
	myowndata.eval_g_python      = g;
	myowndata.eval_jac_g_python  = jacg;
	// logger("D field assigned %p\n", &myowndata);
	// logger("D field assigned %p\n",myowndata.eval_jac_g_python );
		
	if (h !=NULL )
	{
		if (!PyCallable_Check(h))
		{
			PyErr_SetString(PyExc_TypeError, 
					"Need a callable object for function h.");
			return NULL;
		}
		myowndata.eval_h_python	= h;
	}
	else
	{
		logger("[PyIPOPT] Ipopt will use Hessian approximation.\n");
	}

	if (applynew != NULL)
	{
		if (!PyCallable_Check(applynew))
		{
			PyErr_SetString(PyExc_TypeError, 
					"Need a callable object for function applynew.");
			return NULL;
		}
		myowndata.apply_new_python = applynew;
	}
		
	Number* x_L = NULL;                  /* lower bounds on x */
	Number* x_U = NULL;                  /* upper bounds on x */
	Number* g_L = NULL;                  /* lower bounds on g */
	Number* g_U = NULL;                  /* upper bounds on g */
    
	if (n<0) {
		PyErr_SetString(PyExc_ValueError, "Input dimension must be greater than 1");
		return NULL;
	}
	if (m<0) {
		PyErr_SetString(PyExc_ValueError, "Number of constraints be positive or zero");
		return NULL;
	}
			
	x_L = (Number*)malloc(sizeof(Number)*n);
	x_U = (Number*)malloc(sizeof(Number)*n);
	if (!x_L || !x_U)
	{
		PyErr_SetString(PyExc_SystemError, "Cannot allocate memory");
		return NULL;
	}
    
	xldata = (double*)xL->data;
	xudata = (double*)xU->data;
	for (i = 0; i< n; i++) {
		x_L[i] = xldata[i];
		x_U[i] = xudata[i];
	}
		 
	g_L = (Number*)malloc(sizeof(Number)*m);
	g_U = (Number*)malloc(sizeof(Number)*m);		
	if (!g_L || !g_U)
	{
		PyErr_SetString(PyExc_SystemError, "Cannot allocate memory");
		return NULL;
	}
		
	gldata = (double*)gL->data;
	gudata = (double*)gU->data;
		
	for (i = 0; i< m; i++)
	{
		g_L[i] = gldata[i];
		g_U[i] = gudata[i];
	}

	/* create the Ipopt Problem */
	  	
	int C_indexstyle = 0;
	logger("[PyIPOPT] nele_hess is %d\n", nele_hess);
	IpoptProblem thisnlp = CreateIpoptProblem(n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess, C_indexstyle,  &eval_f, &eval_g, &eval_grad_f,  &eval_jac_g, &eval_h);
	logger("[PyIPOPT] Problem created");
		
	// AddIpoptStrOption(thisnlp, "max_iter", 200);
		
	problem *object = NULL;
		
	object = PyObject_NEW(problem , &IpoptProblemType);
	if (!object) return NULL;
		
	object->nlp = thisnlp;
	DispatchData *dp = malloc(sizeof(DispatchData));
	memcpy((void*)dp, (void*)&myowndata, sizeof(DispatchData));
	object->data = dp;
				
      	free(x_L);
	free(x_U);
	free(g_L);
	free(g_U);
	return (PyObject *)object;
}



PyObject *solve(PyObject *self, PyObject *args)
{
    enum ApplicationReturnStatus status; /* Solve return code */
    int i;
    int n;
	
	
  	Number* mult_x_L = NULL;
  	Number* mult_x_U = NULL; 
  	/* Return values */
  	problem* temp = (problem*)self;
  	
  	IpoptProblem nlp = (IpoptProblem)(temp->nlp);
  	DispatchData* bigfield = (DispatchData*)(temp->data);
  	
  	int dX[1];
  	int dL[1];
  	
  	PyArrayObject *x, *mL, *mU;
  	Number obj;                          /* objective value */
  	
  	PyObject* result = Py_False;
	PyArrayObject *x0;
	
	PyObject* myuserdata = NULL;
	
	if (!PyArg_ParseTuple(args, "O!|O", &PyArray_Type, &x0, &myuserdata)) 
    {
		printf("[Error] Parameter X0 is expected to be an Numpy array type.\n");
		Py_INCREF(Py_False);
		return Py_False;
	}
	
	if (myuserdata != NULL)
	{
		bigfield->userdata = myuserdata;
		logger("[PyIPOPT] User specified data field to callback function.\n");
	}
		
	if (nlp == NULL)
	{
		printf("[Error] nlp objective passed to solve is NULL\n Problem created?\n");
		Py_INCREF(Py_False);
		return Py_False;
	}
 	
	/* set some options */
  	
  	// AddIpoptNumOption(nlp, "tol", 1e-8);
  	// AddIpoptStrOption(nlp, "mu_strategy", "adaptive");
  	if (bigfield->eval_h_python == NULL)
  	{
  		AddIpoptStrOption(nlp, "hessian_approximation","limited-memory");
		//logger("Can't find eval_h callback function\n");
	}
  	/* allocate space for the initial point and set the values */
  	
  	// There is a compiler warning here, don't panic, it's correct 
  	int* dim = ((PyArrayObject*)x0)->dimensions; 
  	n = dim[0];
  	dX[0]  = n;
	// logger("n is %d, m is %d\n", n, m);
	x = (PyArrayObject *)PyArray_FromDims( 1, dX, PyArray_DOUBLE );
	
	Number* newx0 = (Number*)malloc(sizeof(Number)*n);
	double* xdata = (double*) x0->data;
	for (i =0; i< n; i++)
		newx0[i] = xdata[i];
	
  	mL = (PyArrayObject *)PyArray_FromDims( 1, dX, PyArray_DOUBLE );
	mU = (PyArrayObject *)PyArray_FromDims( 1, dX, PyArray_DOUBLE );
	// logger("Ready to go\n");
			
  	status = IpoptSolve(nlp, newx0, NULL, &obj, NULL, (double*)mL->data, (double*)mU->data, (UserDataPtr)bigfield);
 	// The final parameter is the userdata (void * type)
 
 	// For status code, see: IpReturnCodes_inc.h 
  	if (status == Solve_Succeeded || Solved_To_Acceptable_Level ) {
  		logger("Problem solved\n");
		double* xdata = (double*) x->data;
		for (i =0; i< n; i++)
			xdata[i] = newx0[i];
			// FreeIpoptProblem(nlp);
		
		if (newx0) free(newx0);
		
		/* A fix for the mem-leak problem */
		return Py_BuildValue( "NNNd",
                              PyArray_Return( x ),
                              PyArray_Return( mL ),
                              PyArray_Return( mU ), obj);
  	}
  	
  	
  	else {
  		// FreeIpoptProblem(nlp);
  		printf("[Error] Ipopt faied in solving problem instance\n");
  		Py_INCREF(Py_False);
  		return Py_False;
	}
}



        
PyObject *close_model(PyObject *self, PyObject *args)
{
	problem* obj = (problem*) self;
	FreeIpoptProblem(obj->nlp);
	obj->nlp = NULL;
	Py_INCREF(Py_True);
	return Py_True;
}

static char PYTEST[] = "TestCreate\n";

static PyObject *test(PyObject *self, PyObject *args)
{
	IpoptProblem thisnlp = NULL;
  	problem *object = NULL;
	object = PyObject_NEW(problem , &IpoptProblemType);
	if (object != NULL)
    	object->nlp = thisnlp;
// 	problem *object = problem_new(thisnlp);
    return (PyObject *)object;
}

/* Begin Python Module code section */
static PyMethodDef ipoptMethods[] = {
 //    { "solve", solve, METH_VARARGS, PYIPOPT_SOLVE_DOC},
    { "create", create, METH_VARARGS, PYIPOPT_CREATE_DOC},
    // { "close",  close_model, METH_VARARGS, PYIPOPT_CLOSE_DOC}, 
   // { "test",   test, 		METH_VARARGS, PYTEST},
    { NULL, NULL }
};

PyMODINIT_FUNC 
initpyipopt(void)
{

	   PyObject* m = 
	   		Py_InitModule3("pyipopt", ipoptMethods, 
	   			"A hooker between Ipopt and Python");
	   
	   import_array( );         /* Initialize the Numarray module. */
		/* A segfault will occur if I use numarray without this.. */

	   if (PyErr_Occurred())	
	 	  Py_FatalError("Unable to initialize module pyipopt");
	 	  
    	return;
}
/* End Python Module code section */

