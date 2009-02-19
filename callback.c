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

// This file contains five call-back functions for IPOPT
/* For the interface of these five functions, check 
	Ipopt's document: Ipopt C Interface */
	// TODO: change the apply_new interface
	
#include "hook.h"

#if 0
void logger(const char* fmt,...)
{
	va_list ap;
	va_start(ap,fmt);
	vprintf(fmt,ap);
	va_end(ap);
        printf("\n");
	fflush(stdout);
}
#else
void logger(const char* fmt,...) {}
#endif


#define Is_double_Array(obj) ((PyArray_TYPE(obj)) == NPY_DOUBLE)
#define Is_long_Array(obj)   ((PyArray_TYPE(obj)) == NPY_LONG)

Bool eval_f(Index n, Number* x, Bool new_x,
            Number* obj_value, UserDataPtr data)
{
	logger("[Callback:E]eval_f");
	int dims[1];
	dims[0] = n;

	DispatchData *myowndata = (DispatchData*) data;
	UserDataPtr user_data = (UserDataPtr) myowndata->userdata;
	
	PyObject *arrayx = PyArray_FromDimsAndData(1, dims, PyArray_DOUBLE , (char*) x);
	if (!arrayx) return FALSE;

	if (new_x && myowndata->apply_new_python) {
		/* Call the python function to applynew */
		PyObject* arg1;
		arg1 = Py_BuildValue("(O)", arrayx);
		PyObject* tempresult = PyObject_CallObject (myowndata->apply_new_python, arg1);
		if (!tempresult) {
			printf("[Error] Python function apply_new returns a None\n");
			Py_DECREF(arg1);	
			return FALSE;
		}
		Py_DECREF(arg1);
		Py_DECREF(tempresult);
	}

	
	PyObject* arglist;
	
	if (user_data != NULL)
		arglist = Py_BuildValue("(OO)", arrayx, (PyObject*)user_data);
	else 
		arglist = Py_BuildValue("(O)", arrayx);

	PyObject* result  = PyObject_CallObject (myowndata->eval_f_python ,arglist);

	if (!result) 
	{
		PyErr_Print();
		Py_DECREF(arrayx);
		Py_CLEAR(arglist);
		return FALSE;
	}
		
	if (!PyFloat_Check(result))
	{
		PyErr_Print();
		Py_DECREF(result);
		Py_DECREF(arrayx);
		Py_CLEAR(arglist);
		return FALSE;
	}
	
	*obj_value =  PyFloat_AsDouble(result);
	Py_DECREF(result);
  	Py_DECREF(arrayx);
	Py_CLEAR(arglist);
	logger("[Callback:R] eval_f");
  	return TRUE;
}

Bool eval_grad_f(Index n, Number* x, Bool new_x,
                 Number* grad_f, UserDataPtr data)
{
	logger("[Callback:E] eval_grad_f");
	
	DispatchData *myowndata = (DispatchData*) data;
	UserDataPtr user_data = (UserDataPtr) myowndata->userdata;
	
	if (myowndata->eval_grad_f_python == NULL) PyErr_Print();
	
	int dims[1];
	dims[0] = n;
	import_array1(FALSE); 
	
	PyObject *arrayx = PyArray_FromDimsAndData(1, dims, PyArray_DOUBLE , (char*) x);
	if (!arrayx) return FALSE;
	
	if (new_x && myowndata->apply_new_python) {
		/* Call the python function to applynew */
		PyObject* arg1 = Py_BuildValue("(O)", arrayx);
		PyObject* tempresult = PyObject_CallObject (myowndata->apply_new_python, arg1);
		if (!tempresult) {
			printf("[Error] Python function apply_new returns a None\n");
			Py_DECREF(arg1);	
			return FALSE;
		}
		Py_DECREF(arg1);
		Py_DECREF(tempresult);
	}	
	
	PyObject* arglist; 
	
	if (user_data != NULL)
		arglist = Py_BuildValue("(OO)", arrayx, (PyObject*)user_data);
	else 
		arglist = Py_BuildValue("(O)", arrayx);
	
	PyArrayObject* result = 
		(PyArrayObject*) PyObject_CallObject 
				(myowndata->eval_grad_f_python, arglist);
	
	if (!result) 
	{
		PyErr_Print();
		Py_DECREF(arrayx);
		Py_CLEAR(arglist);
		return FALSE;
	}
		
	if (!PyArray_Check(result))
	{
		PyErr_Print();
		Py_DECREF(result);
		Py_DECREF(arrayx);
		Py_CLEAR(arglist);
		return FALSE;
	}
#define CHECK(expr,msg)							\
	do if (!(expr))							\
	{								\
		PyErr_SetString(PyExc_TypeError, "eval_grad_f: " msg);	\
		PyErr_Print();						\
		Py_DECREF(result);					\
		Py_DECREF(arrayx);					\
		Py_CLEAR(arglist);					\
		return FALSE;						\
	} while(0)
	CHECK(PyArray_ISCONTIGUOUS(result),"result array must be contiguous");
	CHECK(Is_double_Array(result),"result must be a float array");
	CHECK(1==PyArray_NDIM(result),"result must be a 1d array");
	CHECK(n==PyArray_DIM(result,0),
		"result must have as many elements as the input vector");
#undef CHECK	
	
	double *tempdata = (double*)result->data;
	int i;
	for (i = 0; i < n; i++)
		grad_f[i] = tempdata[i];
		
	Py_DECREF(result);
  	Py_CLEAR(arrayx);
	Py_CLEAR(arglist);
	logger("[Callback:R] eval_grad_f");	
	return TRUE;
}


Bool eval_g(Index n, Number* x, Bool new_x,
            Index m, Number* g, UserDataPtr data)
{

	logger("[Callback:E] eval_g");

	DispatchData *myowndata = (DispatchData*) data;
	UserDataPtr user_data = (UserDataPtr) myowndata->userdata;
	
	if (myowndata->eval_g_python == NULL) 
		PyErr_Print();
	int dims[1];
	int i;
	double *tempdata;
	
	dims[0] = n;
	import_array1(FALSE);
	
	PyObject *arrayx = PyArray_FromDimsAndData(1, dims, PyArray_DOUBLE , (char*) x);
	if (!arrayx) return FALSE;
	
	if (new_x && myowndata->apply_new_python) {
		/* Call the python function to applynew */
		PyObject* arg1 = Py_BuildValue("(O)", arrayx);
		PyObject* tempresult = PyObject_CallObject (myowndata->apply_new_python, arg1);
		if (!tempresult) {
			printf("[Error] Python function apply_new returns a None\n");
			Py_DECREF(arg1);	
			return FALSE;
		}
		Py_DECREF(arg1);
		Py_DECREF(tempresult);
	}
	
	PyObject* arglist; 
	
	if (user_data != NULL)
		arglist = Py_BuildValue("(OO)", arrayx, (PyObject*)user_data);
	else 
	
		arglist = Py_BuildValue("(O)", arrayx);
	
	PyArrayObject* result = 
		(PyArrayObject*) PyObject_CallObject 
			(myowndata->eval_g_python, arglist);
	
	if (!result) 
	{
		PyErr_Print();
		Py_DECREF(arrayx);
		Py_CLEAR(arglist);
		return FALSE;
	}
		
	if (!PyArray_Check(result))
	{
		PyErr_Print();
		Py_DECREF(result);
		Py_DECREF(arrayx);
		Py_CLEAR(arglist);
		return FALSE;
	}
	
#define CHECK(expr,msg)							\
	do if (!(expr))							\
	{								\
		PyErr_SetString(PyExc_TypeError, "eval_g: " msg);	\
		PyErr_Print();						\
		Py_DECREF(result);					\
		Py_DECREF(arrayx);					\
		Py_CLEAR(arglist);					\
		return FALSE;						\
	} while(0)
	CHECK(PyArray_ISCONTIGUOUS(result),"result array must be contiguous");
	CHECK(Is_double_Array(result),"result must be a float array");
	CHECK(1==PyArray_NDIM(result),"result must be a 1d array");
	CHECK(m==PyArray_DIM(result,0),
		"result must have as many elements as constraints");
#undef CHECK	
	tempdata = (double*)result->data;
	for (i = 0; i < m; i++)
		g[i] = tempdata[i];
		
	Py_DECREF(result);
  	Py_CLEAR(arrayx);
	Py_CLEAR(arglist);
	logger("[Callback:R] eval_g");
	return TRUE;
}

Bool eval_jac_g(Index n, Number *x, Bool new_x,
                Index m, Index nele_jac,
                Index *iRow, Index *jCol, Number *values,
                UserDataPtr data)
{

	logger("[Callback:E] eval_jac_g");

	DispatchData *myowndata = (DispatchData*) data;
	UserDataPtr user_data = (UserDataPtr) myowndata->userdata;
	
	int i;
	long* rowd = NULL; 
	long* cold = NULL;
	
	int dims[1];
	dims[0] = n;
	
	double *tempdata;

	if (myowndata->eval_grad_f_python == NULL) 
		PyErr_Print();

	if (values == NULL) {
		import_array1(FALSE);
		PyObject *arrayx = PyArray_FromDimsAndData(1, 
		 			dims, PyArray_DOUBLE , (char*) x);
		if (!arrayx) return FALSE;
		
		PyObject* arglist; 
		
		if (user_data != NULL)
			arglist = Py_BuildValue("(OOO)", 
					arrayx, Py_True, (PyObject*)user_data);
		else 
			arglist = Py_BuildValue("(OO)", arrayx, Py_True);	
		
		PyObject* result = PyObject_CallObject (myowndata->eval_jac_g_python, arglist);
		if (!result)
		{
			PyErr_Print();
			Py_DECREF(arrayx);
			Py_CLEAR(arglist);
			return FALSE;
		}
		PyArrayObject *row = NULL, *col = NULL; 
		if (!PyArg_ParseTuple(result, "O!O!;result of eval_jac_g must be two arrays in a tuple",
				      &PyArray_Type, &row,
				      &PyArray_Type, &col) &&
		    row && col)
		{
			PyErr_Print();
			Py_DECREF(result);
			Py_DECREF(arrayx);
			Py_CLEAR(arglist);
			return FALSE;
		}
#define CHECK(expr,msg) do {						\
		if (!(expr))						\
		{							\
			PyErr_SetString(PyExc_TypeError, "eval_jac_g: " msg); \
			PyErr_Print();					\
			Py_DECREF(result);				\
			Py_DECREF(arrayx);				\
			Py_CLEAR(arglist);				\
			return FALSE;					\
		} } while(0)
		CHECK(PyArray_ISCONTIGUOUS(row),"rows must be contiguous");
		CHECK(PyArray_ISCONTIGUOUS(col),"columns must be contiguous");
		CHECK(Is_long_Array(row),"rows must be an integer array");
		CHECK(Is_long_Array(col),"columns must be an integer array");
		CHECK(1 == PyArray_NDIM(row),"rows must be a 1d array");
		CHECK(1 == PyArray_NDIM(col),"columns must be a 1d array");
		CHECK(nele_jac == PyArray_DIM(row,0),
			"there must be as many rows as non-zero jacobian values");
		CHECK(nele_jac == PyArray_DIM(col,0),
			"there must be as many columns as non-zero jacobian values");
#undef CHECK

		rowd = (long*) row->data;
		cold = (long*) col->data;
		
		for (i = 0; i < nele_jac; i++) {
			iRow[i] = (Index) rowd[i];
			jCol[i] = (Index) cold[i];
			//logger("%d Row %d, Col %d\n", i, iRow[i], jCol[i]);
		}
		Py_CLEAR(arrayx);
		Py_DECREF(result);
		Py_CLEAR(arglist);
		logger("[Callback:R] eval_jac_g(1)");	
	}
	
	else {
		PyObject *arrayx = PyArray_FromDimsAndData(1, 
					dims, PyArray_DOUBLE , (char*) x);
		if (!arrayx) return FALSE;
		
		if (new_x && myowndata->apply_new_python) {
			/* Call the python function to applynew */
			PyObject* arg1 = Py_BuildValue("(O)", arrayx);
			PyObject* tempresult = PyObject_CallObject (myowndata->apply_new_python, arg1);
			if (!tempresult) {
				printf("[Error] Python function apply_new returns a None\n");
				Py_DECREF(arg1);	
				return FALSE;
			}
			Py_DECREF(arg1);
			Py_DECREF(tempresult);
		}
		
		PyObject* arglist; 
		if (user_data != NULL)
			arglist = Py_BuildValue("(OOO)", 
					arrayx, Py_False, (PyObject*)user_data);
		else 
			arglist = Py_BuildValue("(OO)", arrayx, Py_False);	
		
		PyArrayObject* result = (PyArrayObject*) 
					PyObject_CallObject (myowndata->eval_jac_g_python, arglist);
		
		if (!result)
		{
			PyErr_Print();
			Py_DECREF(arrayx);
			Py_CLEAR(arglist);
			return FALSE;
		}
		if (!PyArray_Check(result)) 
		{
			PyErr_Print();
			Py_DECREF(result);
			Py_DECREF(arrayx);
			Py_CLEAR(arglist);
			return FALSE;
		}
#define CHECK(expr,msg)							\
		do if (!(expr))						\
		{							\
			PyErr_SetString(PyExc_TypeError, "eval_jac_g: " msg); \
			PyErr_Print();					\
			Py_DECREF(result);				\
			Py_DECREF(arrayx);				\
			Py_CLEAR(arglist);				\
			return FALSE;					\
		} while(0)
		CHECK(PyArray_ISCONTIGUOUS(result),"result array must be contiguous");
		CHECK(Is_double_Array(result),"result must be a float array");
		CHECK(1==PyArray_NDIM(result),"result must be a 1d array");
		CHECK(nele_jac==PyArray_DIM(result,0),
			"result must have as many values as non-zero constraint jacobian values");
#undef CHECK	
		
		tempdata = (double*)result->data;
		
		for (i = 0; i < nele_jac; i++)
			values[i] = tempdata[i];
		
		Py_DECREF(result);
		Py_CLEAR(arrayx);
		Py_CLEAR(arglist);
		logger("[Callback:R] eval_jac_g(2)");
	}
	logger("[Callback:R] eval_jac_g");
  	return TRUE;
}


Bool eval_h(Index n, Number *x, Bool new_x, Number obj_factor,
            Index m, Number *lambda, Bool new_lambda,
            Index nele_hess, Index *iRow, Index *jCol,
            Number *values, UserDataPtr data)
{
	logger("[Callback:E] eval_h");

	DispatchData *myowndata = (DispatchData*) data;
	UserDataPtr user_data = (UserDataPtr) myowndata->userdata;
	

	int i;
	int dims[1];
	int dims2[1];
	
	if (myowndata->eval_h_python == NULL)
	{	printf("[Error] There is no eval_h assigned");
		return FALSE;
	}
	if (values == NULL) {
		PyObject *newx = Py_True;
		PyObject *objfactor = Py_BuildValue("d", obj_factor);
		PyObject *lagrange = Py_True;
		
		PyObject* arglist;
		
		if (user_data != NULL) 
			arglist =  Py_BuildValue("(OOOOO)", newx, lagrange, objfactor, Py_True, (PyObject*)user_data);
		else 
			arglist =  Py_BuildValue("(OOOO)", newx, lagrange, objfactor, Py_True);
		
		PyObject* result 	= 
			PyObject_CallObject (myowndata->eval_h_python, arglist);
		if (!result)
		{
			PyErr_Print();
			Py_DECREF(objfactor);
			Py_CLEAR(arglist);
			return FALSE;
		}
		PyArrayObject *row = NULL, *col = NULL; 
		if (!PyArg_ParseTuple(result, "O!O!;result of eval_h must be two arrays in a tuple",
				      &PyArray_Type, &row,
				      &PyArray_Type, &col) &&
		    row && col)
		{
			PyErr_Print();
			Py_DECREF(result);
			Py_DECREF(objfactor);
			Py_CLEAR(arglist);
			return FALSE;
		}
#define CHECK(expr,msg) do {						\
		if (!(expr))						\
		{							\
			PyErr_SetString(PyExc_TypeError, "eval_h: " msg); \
			PyErr_Print();					\
			Py_DECREF(result);				\
			Py_DECREF(objfactor);				\
			Py_CLEAR(arglist);				\
			return FALSE;					\
		} } while(0)
		CHECK(PyArray_ISCONTIGUOUS(row),"rows must be contiguous");
		CHECK(PyArray_ISCONTIGUOUS(col),"columns must be contiguous");
		CHECK(Is_long_Array(row),"rows must be an integer array");
		CHECK(Is_long_Array(col),"columns must be an integer array");
		CHECK(1 == PyArray_NDIM(row),"rows must be a 1d array");
		CHECK(1 == PyArray_NDIM(col),"columns must be a 1d array");
		CHECK(nele_hess == PyArray_DIM(row,0),
			"there must be as many rows as non-zero hessian values");
		CHECK(nele_hess == PyArray_DIM(col,0),
			"there must be as many columns as non-zero hessian values");
#undef CHECK

		long* rdata = (long*)row->data;
		long* cdata = (long*)col->data;
		
		// A compiler warning here, from long -> Index(int)
		
		for (i = 0; i < nele_hess; i++) {
			iRow[i] = (Index)rdata[i];
			jCol[i] = (Index)cdata[i];
			if ( n < iRow[i] || n < jCol[i] )
			{
				PyErr_SetString(PyExc_TypeError, "eval_h: "
					"Row or column must be less than "
					"number of input elements");
				PyErr_Print();					
				Py_DECREF(result);				
				Py_DECREF(objfactor);
				Py_CLEAR(arglist);				
				return FALSE;					
			}
			// logger("PyIPOPT_DEBUG %d, %d\n", iRow[i], jCol[i]);
		}

		Py_DECREF(objfactor);
		Py_DECREF(result);
		Py_CLEAR(arglist);
		logger("[Callback:R] eval_h (1)");
	}
	else {	
		PyObject *objfactor = Py_BuildValue("d", obj_factor);
		
		dims[0] = n;
		PyObject *arrayx = 
			PyArray_FromDimsAndData(1, dims, PyArray_DOUBLE , (char*) x);
		if (!arrayx) return FALSE;
		
		if (new_x && myowndata->apply_new_python) {
			//  Call the python function to applynew 
			PyObject* arg1 = Py_BuildValue("(O)", arrayx);
			PyObject* tempresult = 
				PyObject_CallObject (myowndata->apply_new_python, arg1);
			if (!tempresult) {
				printf("[Error] Python function apply_new returns a None\n");
				Py_DECREF(arg1);	
				return FALSE;
			}
			Py_DECREF(arg1);
			Py_DECREF(tempresult);
		}
		
		dims2[0] = m;
		PyObject *lagrangex = 
			PyArray_FromDimsAndData(1, dims2, PyArray_DOUBLE , (char*) lambda);
		if (!lagrangex) return FALSE;
		
		PyObject* arglist;
		
		if (user_data != NULL)
			arglist = Py_BuildValue("(OOOOO)", arrayx, lagrangex, objfactor, Py_False, (PyObject*)user_data);
		else
			arglist = Py_BuildValue("(OOOO)", arrayx, lagrangex, objfactor, Py_False);
		PyArrayObject* result = 
			(PyArrayObject*) PyObject_CallObject 
				(myowndata->eval_h_python, arglist);
		
		if (!result)
		{
			PyErr_Print();
			Py_DECREF(arrayx);
			Py_CLEAR(lagrangex);
			Py_CLEAR(objfactor);
			Py_CLEAR(arglist);
			return FALSE;
		}
		if (!PyArray_Check(result)) 
		{
			PyErr_Print();
			Py_DECREF(result);
			Py_CLEAR(lagrangex);
			Py_CLEAR(objfactor);
			Py_DECREF(arrayx);
			Py_CLEAR(arglist);
			return FALSE;
		}
#define CHECK(expr,msg)							\
		do if (!(expr))						\
		{							\
			PyErr_SetString(PyExc_TypeError, "eval_h: " msg); \
			PyErr_Print();					\
			Py_DECREF(result);				\
			Py_CLEAR(lagrangex);				\
			Py_CLEAR(objfactor);				\
			Py_DECREF(arrayx);				\
			Py_CLEAR(arglist);				\
			return FALSE;					\
		} while(0)
		CHECK(PyArray_ISCONTIGUOUS(result),"result array must be contiguous");
		CHECK(Is_double_Array(result),"result must be a float array");
		CHECK(1==PyArray_NDIM(result),"result must be a 1d array");
		CHECK(nele_hess==PyArray_DIM(result,0),
			"result must have as many values as non-zero hessian values");
#undef CHECK	
		
		double* tempdata = (double*)result->data;
		for (i = 0; i < nele_hess; i++)
		{	values[i] = tempdata[i];
			// logger("PyDebug %f \n", values[i]);
		}	
		Py_CLEAR(arrayx);
		Py_CLEAR(lagrangex);
		Py_CLEAR(objfactor);
		Py_DECREF(result);
		Py_CLEAR(arglist);
		logger("[Callback:R] eval_h (2)");
	}	                         
  	return TRUE;
}


