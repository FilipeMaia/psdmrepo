//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataSpectrometerV0...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataSpectrometerV0.h"

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
  PyObject* hproj( PyObject* self, PyObject* );
  PyObject* vproj( PyObject* self, PyObject* );

  PyMethodDef methods[] = {
    {"hproj",      hproj,       METH_NOARGS,  "self.hproj() -> numpy.ndarray\n\nReturns horizontal projection array" },
    {"vproj",      vproj,       METH_NOARGS,  "self.vproj() -> numpy.ndarray\n\nReturns vertical projection array" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Bld::BldDataSpectrometerV0 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Bld::BldDataSpectrometerV0::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "BldDataSpectrometerV0", module );
}

void
pypdsdata::Bld::BldDataSpectrometerV0::print(std::ostream& out) const
{
  if(not m_obj) {
    out << typeName() << "(None)";
  } else {
    out << typeName() << "(hproj=" << m_obj->hproj()
        << ", vproj=" << m_obj->vproj() << ")";
  }
}

namespace {

PyObject*
nd2nd(PyObject* self, const ndarray<const uint32_t, 1>& arr)
{
  // dimensions
  npy_intp dims[1] = { arr.size() };

  // NumPy type number
  int typenum = NPY_UINT32 ;
  int flags = NPY_C_CONTIGUOUS ;

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 2, dims, typenum, 0,
                                (void*)arr.data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

PyObject*
hproj( PyObject* self, PyObject* )
{
  Pds::Bld::BldDataSpectrometerV0* obj = pypdsdata::Bld::BldDataSpectrometerV0::pdsObject( self );
  if ( not obj ) return 0;

  return nd2nd(self, obj->hproj());
}

PyObject*
vproj( PyObject* self, PyObject* )
{
  Pds::Bld::BldDataSpectrometerV0* obj = pypdsdata::Bld::BldDataSpectrometerV0::pdsObject( self );
  if ( not obj ) return 0;

  return nd2nd(self, obj->vproj());
}

}
