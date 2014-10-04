#include <Python.h>
#include <numpy/arrayobject.h>

/* Definition of plain 'C' scaleByN function */
static void ScaleByN(double *array, int arraySize, double scale)  
{  
  int index;
  for (index=0; index<arraySize; index++)  
    {
      array[index] *= scale;
    }

  return;
}


/* PYTHON documentation string for scaleByN function */
char pypsalg_c_scaleByN_docstring[] =
  "Multiply each element of input array by scale";


/* Definition of PYTHON scaleByN function */
PyObject *pypsalg_c_scaleByN(PyObject *self, PyObject *args)
{
  double scale;
  PyObject *array_obj;
  
  /* Parse the input ntuple */
  if (!PyArg_ParseTuple(args, "Od", &array_obj, &scale))
    return NULL;

  /* Interpret array_obj as numpy array */
  PyObject *array = PyArray_FROM_OTF(array_obj, NPY_DOUBLE, NPY_IN_ARRAY);

  /* If that didn't work, throw an exception */
  if (array == NULL) {
    Py_DECREF(array);
    return NULL;
  }

  /* Get array size and acces to array data */
  int arraySize = (int)PyArray_DIM(array, 0);
  double *arrayData = (double*)PyArray_DATA(array);

  /* Call ScaleByN */
  ScaleByN(arrayData, arraySize, scale);

  /* Clean Up */
  Py_DECREF(array);
  
  /* Return NONE */
  Py_RETURN_NONE;
}
