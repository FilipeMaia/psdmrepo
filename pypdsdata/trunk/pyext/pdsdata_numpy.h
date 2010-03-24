#ifndef PYPDSDATA_PDSDATA_NUMPY_H
#define PYPDSDATA_PDSDATA_NUMPY_H
//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Import of the numpy API. All regular clients of numpy need to 
//      include this file. In one place exactly this file needs to be 
//      imported after defining PDSDATA_IMPORT_ARRAY and call import_array()
//
//------------------------------------------------------------------------

#define PY_ARRAY_UNIQUE_SYMBOL pdsdata_ARRAY_API

#ifndef PDSDATA_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif

#include "numpy/arrayobject.h"

#endif // PYPDSDATA_PDSDATA_NUMPY_H
