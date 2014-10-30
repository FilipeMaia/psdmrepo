#include <Python.h>
#include <numpy/arrayobject.h>


/* The PYTHON chi2 function and documenation string */
PyObject *pypsalg_c_chi2(PyObject *self, PyObject *args);
char pypsalg_c_chi2_docstring[];


/* The PYTHON scaleByN function and documentation string */
PyObject *pypsalg_c_scaleByN(PyObject *self, PyObject *args);
char pypsalg_c_scaleByN_docstring[];


/* Array that decalares the PYTHON functions that c_algos package will
 * contain 
 */
static PyMethodDef module_methods[] = {
  {"chi2", pypsalg_c_chi2, METH_VARARGS, pypsalg_c_chi2_docstring},
  {"scaleByN", pypsalg_c_scaleByN, METH_VARARGS, pypsalg_c_scaleByN_docstring},
  {NULL, NULL, 0, NULL}
};






/* Documentation string for the PYTHON c_algos package */
static char module_docstring[] = 
  "This module is the PYTHON interface to algorithms coded in C";

/* Initialisation function for c_algos PYTHON package */
PyMODINIT_FUNC initpypsalg_c(void) 
{
  PyObject *m = Py_InitModule3("pypsalg_c", module_methods, module_docstring);
  if (m == NULL) 
    return;

  /* Load `numpy` functionality */
  import_array();

}





