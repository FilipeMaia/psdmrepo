//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class IpmFexConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "IpmFexConfigV2.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "DiodeFexConfigV2.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Lusi::IpmFexConfigV2, xscale)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Lusi::IpmFexConfigV2, yscale)
  PyObject* IpmFexConfigV2_diode( PyObject* self, void* );

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"xscale",   xscale,                0, "Floating point number", 0},
    {"yscale",   yscale,                0, "Floating point number", 0},
    {"diode",    IpmFexConfigV2_diode,  0, "List of NCHANNELS :py:class:`DiodeFexConfigV2` objects", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Lusi::IpmFexConfigV2 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Lusi::IpmFexConfigV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyObject* val = PyInt_FromLong(Pds::Lusi::IpmFexConfigV2::NCHANNELS);
  PyDict_SetItemString( type->tp_dict, "NCHANNELS", val );
  Py_XDECREF(val);

  BaseType::initType( "IpmFexConfigV2", module );
}

void
pypdsdata::Lusi::IpmFexConfigV2::print(std::ostream& str) const
{
  str << "lusi.IpmFexConfigV2(xscale=" << m_obj->xscale() << ", yscale=" << m_obj->yscale() << ", ...)";
}

namespace {

PyObject*
IpmFexConfigV2_diode( PyObject* self, void* )
{
  Pds::Lusi::IpmFexConfigV2* obj = pypdsdata::Lusi::IpmFexConfigV2::pdsObject(self);
  if (not obj) return 0;

  const int size = Pds::Lusi::IpmFexConfigV2::NCHANNELS;
  PyObject* list = PyList_New( size );
  for ( int i = 0 ; i < size ; ++ i ) {
    Pds::Lusi::DiodeFexConfigV2& dconf = const_cast<Pds::Lusi::DiodeFexConfigV2&>(obj->diode()[i]);
    PyObject* obj = pypdsdata::Lusi::DiodeFexConfigV2::PyObject_FromPds(&dconf, self, sizeof(Pds::Lusi::DiodeFexConfigV2));
    PyList_SET_ITEM( list, i, obj );
  }
  return list;
}

}
