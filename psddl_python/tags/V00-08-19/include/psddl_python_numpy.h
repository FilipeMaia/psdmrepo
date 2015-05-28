#ifndef PSDDL_PYTHON_PSDDL_PYTHON_NUMPY_H
#define PSDDL_PYTHON_PSDDL_PYTHON_NUMPY_H
//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Import of the numpy API. All regular clients of numpy need to 
//      include this file. In one place exactly this file needs to be 
//      imported after defining PSDDL_NUMPY_IMPORT_ARRAY and call _import_array()
//
//------------------------------------------------------------------------

#define PY_ARRAY_UNIQUE_SYMBOL psddl_python_ARRAY_API

#ifndef PSDDL_PYTHON_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif

#include "numpy/arrayobject.h"

#endif // PSDDL_PYTHON_PSDDL_PYTHON_NUMPY_H
