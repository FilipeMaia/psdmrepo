//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

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

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::Arraychar::DataV1, numChars)
  PyObject* data( PyObject* self, PyObject* );

  PyMethodDef methods[] = {
    { "numChars",     numChars,     METH_NOARGS,  "self.numChars() -> int\n\nReturns number of bytes in an array" },
    { "data",         data,         METH_NOARGS,  "self.data() -> ndarray\n\nReturns array of bytes" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Arraychar::DataV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Arraychar::DataV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "DataV1", module );
}

void
pypdsdata::Arraychar::DataV1::print(std::ostream& str) const
{
  str << "Arraychar.DataV1(numChars=" << m_obj->numChars()
      << ", data=" << m_obj->data()
      << ")";
}

namespace {

PyObject*
data( PyObject* self, PyObject* )
{
  Pds::Arraychar::DataV1* obj = pypdsdata::Arraychar::DataV1::pdsObject( self );
  if ( not obj ) return 0;

  // size of data array
  npy_intp dims[1] = { obj->numChars() };

  // NumPy type number
  int typenum = NPY_UBYTE ;
  int flags = NPY_C_CONTIGUOUS ;

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 1, dims, typenum, 0,
                                (void*)obj->data().data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

}
