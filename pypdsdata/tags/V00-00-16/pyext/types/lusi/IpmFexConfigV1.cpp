//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: IpmFexConfigV1.cpp 811 2010-03-26 17:40:08Z salnikov $
//
// Description:
//	Class IpmFexConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "IpmFexConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "DiodeFexConfigV1.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  PyObject* __repr__( PyObject *self );

  // methods
  MEMBER_WRAPPER(pypdsdata::Lusi::IpmFexConfigV1, xscale)
  MEMBER_WRAPPER(pypdsdata::Lusi::IpmFexConfigV1, yscale)
  PyObject* IpmFexConfigV1_diode( PyObject* self, void* );

  PyGetSetDef getset[] = {
    {"xscale",   xscale,                0, "", 0},
    {"yscale",   yscale,                0, "", 0},
    {"diode",    IpmFexConfigV1_diode,  0, "", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Lusi::IpmFexConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Lusi::IpmFexConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = __repr__;
  type->tp_repr = __repr__;

  BaseType::initType( "IpmFexConfigV1", module );
}


namespace {

PyObject*
IpmFexConfigV1_diode( PyObject* self, void* )
{
  pypdsdata::Lusi::IpmFexConfigV1* py_this = static_cast<pypdsdata::Lusi::IpmFexConfigV1*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  const int size = Pds::Lusi::IpmFexConfigV1::NCHANNELS;
  PyObject* list = PyList_New( size );
  for ( int i = 0 ; i < size ; ++ i ) {
    Pds::Lusi::DiodeFexConfigV1& dconf = py_this->m_obj->diode[i];
    PyObject* obj = pypdsdata::Lusi::DiodeFexConfigV1::PyObject_FromPds(&dconf, self, sizeof(Pds::Lusi::DiodeFexConfigV1));
    PyList_SET_ITEM( list, i, obj );
  }
  return list;
}

PyObject*
__repr__( PyObject *self )
{
  pypdsdata::Lusi::IpmFexConfigV1* py_this = static_cast<pypdsdata::Lusi::IpmFexConfigV1*>(self);

  char buf[80];
  snprintf( buf, sizeof buf, "Lusi.IpmFexConfigV1(xscale=%g, yscale=%g, diode=[...])", 
      py_this->m_obj->xscale, py_this->m_obj->yscale);
  return PyString_FromString( buf );
}

}
