//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2GainMapCfg...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CsPad2x2GainMapCfg.h"

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
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"map",     map,      METH_NOARGS, "Returns gain map as an array." },
    {0, 0, 0, 0}
   };

  PyGetSetDef getset[] = {
    {"gainMap",         gainMap,         0, "", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad2x2::CsPad2x2GainMapCfg class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad2x2::CsPad2x2GainMapCfg::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_getset = ::getset;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "CsPad2x2GainMapCfg", module );
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
  const Pds::CsPad2x2::CsPad2x2GainMapCfg* obj = pypdsdata::CsPad2x2::CsPad2x2GainMapCfg::pdsObject( self );
  if ( not obj ) return 0;

  // NumPy type number
  int typenum = NPY_USHORT;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  // dimensions
  npy_intp dims[2] = { Pds::CsPad2x2::ColumnsPerASIC, Pds::CsPad2x2::MaxRowsPerASIC };

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 2, dims, typenum, 0,
                                (void*)obj->_gainMap, 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

PyObject*
_repr( PyObject *self )
{
  const Pds::CsPad2x2::CsPad2x2GainMapCfg* pdsObj = pypdsdata::CsPad2x2::CsPad2x2GainMapCfg::pdsObject( self );
  if(not pdsObj) return 0;

  std::ostringstream str;
  str << "cspad2x2.CsPad2x2GainMapCfg([" << pdsObj->_gainMap[0][0]
      << ", " << pdsObj->_gainMap[0][1]
      << ", " << pdsObj->_gainMap[0][2]
      << ", " << pdsObj->_gainMap[0][3]
      << ", ...])";
  return PyString_FromString( str.str().c_str() );
}

}
