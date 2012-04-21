//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: DiodeFexConfigV1.cpp 811 2010-03-26 17:40:08Z salnikov $
//
// Description:
//	Class DiodeFexConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DiodeFexConfigV1.h"

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
  PyObject* DiodeFexConfigV1_base( PyObject* self, void* );
  PyObject* DiodeFexConfigV1_scale( PyObject* self, void* );

  PyGetSetDef getset[] = {
    {"base",       DiodeFexConfigV1_base,   0, "List of NRANGES floating numbers", 0},
    {"scale",      DiodeFexConfigV1_scale,  0, "List of NRANGES floating numbers", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Lusi::DiodeFexConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Lusi::DiodeFexConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyObject* val = PyInt_FromLong(Pds::Lusi::DiodeFexConfigV1::NRANGES);
  PyDict_SetItemString( type->tp_dict, "NRANGES", val );
  Py_XDECREF(val);

  BaseType::initType( "DiodeFexConfigV1", module );
}


namespace {

PyObject*
DiodeFexConfigV1_base( PyObject* self, void* )
{
  Pds::Lusi::DiodeFexConfigV1* obj = pypdsdata::Lusi::DiodeFexConfigV1::pdsObject(self);
  if (not obj) return 0;

  const int size = Pds::Lusi::DiodeFexConfigV1::NRANGES;
  PyObject* list = PyList_New( size );
  for ( int i = 0 ; i < size ; ++ i ) {
    PyList_SET_ITEM( list, i, pypdsdata::TypeLib::toPython(obj->base[i]) );
  }
  return list;
}

PyObject*
DiodeFexConfigV1_scale( PyObject* self, void* )
{
  Pds::Lusi::DiodeFexConfigV1* obj = pypdsdata::Lusi::DiodeFexConfigV1::pdsObject(self);
  if (not obj) return 0;

  const int size = Pds::Lusi::DiodeFexConfigV1::NRANGES;
  PyObject* list = PyList_New( size );
  for ( int i = 0 ; i < size ; ++ i ) {
    PyList_SET_ITEM( list, i, pypdsdata::TypeLib::toPython(obj->scale[i]) );
  }
  return list;
}

PyObject*
_repr( PyObject *self )
{
  Pds::Lusi::DiodeFexConfigV1* obj = pypdsdata::Lusi::DiodeFexConfigV1::pdsObject(self);
  if (not obj) return 0;

  std::ostringstream str ;
  str << "lusi.DiodeFexConfigV1(base=[" ;
  
  const int size = Pds::Lusi::DiodeFexConfigV1::NRANGES;
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
