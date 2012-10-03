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
  PyObject* _repr( PyObject *self );

  // methods
  MEMBER_WRAPPER(pypdsdata::Lusi::IpmFexConfigV1, xscale)
  MEMBER_WRAPPER(pypdsdata::Lusi::IpmFexConfigV1, yscale)
  PyObject* IpmFexConfigV1_diode( PyObject* self, void* );

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"xscale",   xscale,                0, "Floating point number", 0},
    {"yscale",   yscale,                0, "Floating point number", 0},
    {"diode",    IpmFexConfigV1_diode,  0, "List of NCHANNELS :py:class:`DiodeFexConfigV1` objects", 0},
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
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyObject* val = PyInt_FromLong(Pds::Lusi::IpmFexConfigV1::NCHANNELS);
  PyDict_SetItemString( type->tp_dict, "NCHANNELS", val );
  Py_XDECREF(val);

  BaseType::initType( "IpmFexConfigV1", module );
}


namespace {

PyObject*
IpmFexConfigV1_diode( PyObject* self, void* )
{
  Pds::Lusi::IpmFexConfigV1* obj = pypdsdata::Lusi::IpmFexConfigV1::pdsObject(self);
  if (not obj) return 0;

  const int size = Pds::Lusi::IpmFexConfigV1::NCHANNELS;
  PyObject* list = PyList_New( size );
  for ( int i = 0 ; i < size ; ++ i ) {
    Pds::Lusi::DiodeFexConfigV1& dconf = obj->diode[i];
    PyObject* obj = pypdsdata::Lusi::DiodeFexConfigV1::PyObject_FromPds(&dconf, self, sizeof(Pds::Lusi::DiodeFexConfigV1));
    PyList_SET_ITEM( list, i, obj );
  }
  return list;
}

PyObject*
_repr( PyObject *self )
{
  Pds::Lusi::IpmFexConfigV1* obj = pypdsdata::Lusi::IpmFexConfigV1::pdsObject(self);
  if (not obj) return 0;

  char buf[80];
  snprintf( buf, sizeof buf, "lusi.IpmFexConfigV1(xscale=%g, yscale=%g, ...)", 
      obj->xscale, obj->yscale);
  return PyString_FromString( buf );
}

}
