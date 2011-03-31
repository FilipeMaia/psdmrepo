//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadGainMapCfg...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "CsPadGainMapCfg.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  PyObject* map( PyObject* self, PyObject* );
  PyObject* gainMap( PyObject* self, void* );

  PyMethodDef methods[] = {
    {"map",     map,      METH_NOARGS, "Returns gain map as an array." },
    {0, 0, 0, 0}
   };

  PyGetSetDef getset[] = {
    {"gainMap",         gainMap,         0, "", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad::CsPadGainMapCfg class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad::CsPadGainMapCfg::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_getset = ::getset;

  BaseType::initType( "CsPadGainMapCfg", module );
}

namespace {

PyObject*
map( PyObject* self, PyObject* )
{
  return gainMap(self, 0);
}
PyObject*
gainMap( PyObject* self, void* )
{
  const Pds::CsPad::CsPadGainMapCfg* obj = pypdsdata::CsPad::CsPadGainMapCfg::pdsObject( self );
  if ( not obj ) return 0;

  // NumPy type number
  int typenum = NPY_USHORT;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  // dimensions
  npy_intp dims[2] = { Pds::CsPad::ColumnsPerASIC, Pds::CsPad::MaxRowsPerASIC };

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 2, dims, typenum, 0,
                                (void*)obj->_gainMap, 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

}
