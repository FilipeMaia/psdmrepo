//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DiodeFexConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DiodeFexConfigV2.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  PyObject* _repr( PyObject *self );

  // methods
  PyObject* DiodeFexConfigV2_base( PyObject* self, void* );
  PyObject* DiodeFexConfigV2_scale( PyObject* self, void* );

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"base",       DiodeFexConfigV2_base,   0, "List of NRANGES floating numbers", 0},
    {"scale",      DiodeFexConfigV2_scale,  0, "List of NRANGES floating numbers", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Lusi::DiodeFexConfigV2 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Lusi::DiodeFexConfigV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyObject* val = PyInt_FromLong(Pds::Lusi::DiodeFexConfigV2::NRANGES);
  PyDict_SetItemString( type->tp_dict, "NRANGES", val );
  Py_XDECREF(val);

  BaseType::initType( "DiodeFexConfigV2", module );
}


namespace {

PyObject*
DiodeFexConfigV2_base( PyObject* self, void* )
{
  Pds::Lusi::DiodeFexConfigV2* obj = pypdsdata::Lusi::DiodeFexConfigV2::pdsObject(self);
  if (not obj) return 0;

  const int size = Pds::Lusi::DiodeFexConfigV2::NRANGES;
  PyObject* list = PyList_New( size );
  for ( int i = 0 ; i < size ; ++ i ) {
    PyList_SET_ITEM( list, i, pypdsdata::TypeLib::toPython(obj->base[i]) );
  }
  return list;
}

PyObject*
DiodeFexConfigV2_scale( PyObject* self, void* )
{
  Pds::Lusi::DiodeFexConfigV2* obj = pypdsdata::Lusi::DiodeFexConfigV2::pdsObject(self);
  if (not obj) return 0;

  const int size = Pds::Lusi::DiodeFexConfigV2::NRANGES;
  PyObject* list = PyList_New( size );
  for ( int i = 0 ; i < size ; ++ i ) {
    PyList_SET_ITEM( list, i, pypdsdata::TypeLib::toPython(obj->scale[i]) );
  }
  return list;
}

PyObject*
_repr( PyObject *self )
{
  Pds::Lusi::DiodeFexConfigV2* obj = pypdsdata::Lusi::DiodeFexConfigV2::pdsObject(self);
  if (not obj) return 0;

  std::ostringstream str ;
  str << "lusi.DiodeFexConfigV2(base=[" ;
  
  const int size = Pds::Lusi::DiodeFexConfigV2::NRANGES;
  for ( int i = 0 ; i < size ; ++ i ) {
    if ( i ) str << ", " ;
    str << obj->base[i];
  }
  str << "], scale=[" ;
  for ( int i = 0 ; i < size ; ++ i ) {
    if ( i ) str << ", " ;
    str << obj->scale[i];
  }
  str << "])" ;

  return PyString_FromString( str.str().c_str() );
}

}
