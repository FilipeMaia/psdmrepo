//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EnumType...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "EnumType.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // enum object is an integer with attached name
  struct EnumObject : PyIntObject {
    PyObject* en_name ;
  };

  PyObject initHeader = { PyObject_HEAD_INIT(0) };

  char docString[] =
    "Python class which emulates C++ enum type. It defines class\n"
    "attributes whose values are integer numbers, so that you can\n"
    "use EnumType.EnumValue notation to access the values. Instances\n"
    "of this class are integer numbers which when converted to string\n"
    "give the name of the corresponding class attribute.";

  // standard Python stuff
  void Enum_dealloc( PyObject* self );
  PyObject* Enum_str_repr( PyObject* self );
  int Enum_init(PyObject* self, PyObject* args, PyObject* kwds);

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

//----------------
// Constructors --
//----------------
pypdsdata::EnumType::EnumType ( const char* name, Enum* enums )
  : m_type()
  , m_int2enum()
{

  // initialize type structure
  memset ( &m_type, 0, sizeof m_type );
  memcpy( &m_type, &initHeader, sizeof initHeader );
  m_type.tp_name = (char*)name;
  m_type.tp_basicsize = sizeof(EnumObject);
  m_type.tp_dealloc = Enum_dealloc;
  m_type.tp_repr = Enum_str_repr;
  m_type.tp_str = Enum_str_repr;
  m_type.tp_flags = Py_TPFLAGS_DEFAULT;
  m_type.tp_doc = ::docString ;
  m_type.tp_base = &PyInt_Type ;
  m_type.tp_init = Enum_init ;
  m_type.tp_alloc = PyType_GenericAlloc ;
  m_type.tp_new = PyType_GenericNew ;
  m_type.tp_free = _PyObject_Del ;
  m_type.tp_del = Enum_dealloc;

  PyObject* tp_dict = PyDict_New();
  m_type.tp_dict = tp_dict;

  // finalize
  PyType_Ready( &m_type );

  // this enum class name
  const char* p = strrchr( m_type.tp_name, '.' );
  std::string type = p ? p+1 : m_type.tp_name;
  type += '.';

  // define class attributes as enum values
  for ( ; enums->name ; ++ enums ) {

    // build the object
    EnumObject* value = PyObject_New(EnumObject, &m_type);
    std::string name = type + enums->name;
    value->ob_ival = enums->value;
    value->en_name = PyString_FromString( name.c_str() );

    // store it in the class attribute
    PyObject* key = PyString_FromString(enums->name) ;
    PyDict_SetItem( m_type.tp_dict, key, (PyObject*)value );
    Py_DECREF(key);

    // and in the map
    m_int2enum[enums->value] = (PyObject*)value;
  }

}

//--------------
// Destructor --
//--------------
pypdsdata::EnumType::~EnumType ()
{
  // "free" all referenced objects
  for ( Int2Enum::iterator it = m_int2enum.begin() ; it != m_int2enum.begin() ; ++ it ) {
    Py_CLEAR( it->second );
  }
}


// Make instance of this type
PyObject*
pypdsdata::EnumType::Enum_FromLong( long value )
{
  Int2Enum::const_iterator it = m_int2enum.find( value );
  if ( it == m_int2enum.end() ) {
    PyErr_Format( PyExc_RuntimeError, "Unknown enum value (%ld)", value );
    return 0;
  }

  Py_INCREF( it->second );
  return it->second;
}

PyObject*
pypdsdata::EnumType::Enum_FromString( const char* name )
{
  // try to find a name in the class attributes
  if ( not m_type.tp_dict ) {
    PyErr_SetString( PyExc_TypeError, "Type has no dictionary." );
    return 0;
  }

  // get the class attribute with this name
  PyObject* val = PyDict_GetItemString( m_type.tp_dict, name );

  if ( not val ) {
    PyErr_Format( PyExc_TypeError, "Unknown enum name (%s)", name );
    return 0;
  }

  // return the object
  Py_INCREF( val );
  return val;
}

namespace {

int
Enum_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  EnumObject* py_this = (EnumObject*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }
  py_this->ob_ival = 0;
  py_this->en_name = 0;

  // expect integer or string
  PyObject* arg;
  if ( not PyArg_ParseTuple( args, "O:Enum_Init", &arg ) ) {
    return -1 ;
  }

  if ( PyInt_Check(arg) ) {

    // dirty hack
    pypdsdata::EnumType* enumType = (pypdsdata::EnumType*)self->ob_type;
    if ( PyObject* o = enumType->Enum_FromLong( PyInt_AsLong(arg) ) ) {

      EnumObject* enumObj = (EnumObject*)o;
      py_this->ob_ival = enumObj->ob_ival;
      py_this->en_name = enumObj->en_name;
      Py_INCREF(py_this->en_name);

    } else {

      return -1;

    }

  } else if ( PyString_Check(arg) ) {

    // dirty hack
    pypdsdata::EnumType* enumType = (pypdsdata::EnumType*)self->ob_type;
    if ( PyObject* o = enumType->Enum_FromString( PyString_AsString(arg) ) ) {

      EnumObject* enumObj = (EnumObject*)o;
      py_this->ob_ival = enumObj->ob_ival;
      py_this->en_name = enumObj->en_name;
      Py_INCREF(py_this->en_name);

    } else {

      return -1;

    }

  } else {

    PyErr_SetString(PyExc_RuntimeError, "Error: unknown argument type");
    return -1;

  }

  return 0;
}

void
Enum_dealloc( PyObject* self )
{
  // free the name
  Py_CLEAR( ((EnumObject*)self)->en_name ) ;

  // deallocate ourself
  self->ob_type->tp_free(self);
}

PyObject*
Enum_str_repr( PyObject* self )
{
  PyObject* name = ((EnumObject*) self)->en_name;
  Py_INCREF(name) ;
  return name;
}

}
