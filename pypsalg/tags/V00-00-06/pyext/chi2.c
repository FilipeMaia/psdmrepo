#include <Python.h>
#include <numpy/arrayobject.h>

/* Definition of plain 'C' chi2 function */
static double chi2(double m, double b, double *x, double *y, double *yerr, int N) {
  int n;
  double result=0.0, diff;

  for (n = 0; n < N; n++) {
    diff = (y[n] - (m * x[n] + b)) / yerr[n];
    result += diff * diff;      
  }

  return result;
}


/* PYTHON documentation string for chi2 function */
char pypsalg_c_chi2_docstring[] =
  "Calculate the chi-squared of some data given a model.";


/* Definition of PYTHON chi2 function */
PyObject *pypsalg_c_chi2(PyObject *self, PyObject *args) 
{
  double m, b;
  PyObject *x_obj, *y_obj, *yerr_obj;

  /* Parse the input ntuple */
  if (!PyArg_ParseTuple(args, "ddOOO", &m, &b, &x_obj, &y_obj, &yerr_obj))
    return NULL;

  /* Interpret the input objects as numpy arrays */
  PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  PyObject *yerr_array = PyArray_FROM_OTF(yerr_obj, NPY_DOUBLE, NPY_IN_ARRAY);

  /* If that didn't work, throw an exception */
  if (x_array == NULL || y_array == NULL || yerr_array == NULL) {
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(yerr_array);
    return NULL;
  }
     
  /* How many data points are there? */
  int N = (int)PyArray_DIM(x_array, 0);

  /* Get pointers to the data as C-types */
  double *x = (double*)PyArray_DATA(x_array);
  double *y = (double*)PyArray_DATA(y_array);
  double *yerr = (double*)PyArray_DATA(yerr_array);

  /* Call the external C function to compute the chi-squared. */
  double value = chi2(m, b, x, y, yerr, N);

  /* Clean up */
  Py_DECREF(x_array);
  Py_DECREF(y_array);
  Py_DECREF(yerr_array);

  if (value < 0.0) {
    PyErr_SetString(PyExc_RuntimeError,
		    "Chi-squared returned an impossible value.");
    return NULL;
  }

  /* Build the output tuple */
  PyObject *ret = Py_BuildValue("d", value);
  return ret;
}


