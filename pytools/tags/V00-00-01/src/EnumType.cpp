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
#include "pytools/EnumType.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <cstring>
#include <sstream>

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
    "Python class which emulates C++ enumeration. It defines class attributes\n"
    "whose values are integer numbers, so that you can use EnumType.EnumValue\n"
    "notation to access the values. Instances of this class are integer numbers\n"
    "which when converted to string give the name of the corresponding class\n"
    "attribute, e.g. str(Enum.SomeConst) => \"Enum.SomeConst\" (but \n"
    "repr(Enum.SomeConst) => 42)\n";

  // standard Python stuff
  void Enum_dealloc( PyObject* self );
  PyObject* Enum_str_repr( PyObject* self );
  PyObject* Enum_str_int( PyObject* self );
  int Enum_init(PyObject* self, PyObject* args, PyObject* kwds);

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

//----------------
// Constructors --
//----------------
pytools::EnumType::EnumType(const char* typeName)
  : m_typeName(0)
  , m_type()
  , m_int2enum()
  , m_docString(0)
{
  initType(typeName);

  // generate doc string
  makeDocString();
}

pytools::EnumType::EnumType(const char* typeName, Enum* enums)
  : m_typeName(0)
  , m_type()
  , m_int2enum()
  , m_docString(0)
{
  initType(typeName);

  // this enum class name
  const char* p = strrchr( m_type.tp_name, '.' );
  std::string type = p ? p+1 : m_type.tp_name;
  type += '.';

  // define class attributes as enum values
  for (Enum* eiter = enums; eiter->name ; ++ eiter) {

    // build the object
    EnumObject* value = PyObject_New(EnumObject, &m_type);
    std::string name = type + eiter->name;
    value->ob_ival = eiter->value;
    value->en_name = PyString_FromString(name.c_str());

    // store it in the class attribute
    PyDict_SetItemString( m_type.tp_dict, eiter->name, (PyObject*)value );

    // and in the map
    m_int2enum[eiter->value] = (PyObject*)value;
  }

  // generate doc string
  makeDocString();
}

//--------------
// Destructor --
//--------------
pytools::EnumType::~EnumType ()
{
  // "free" all referenced objects
  for ( Int2Enum::iterator it = m_int2enum.begin() ; it != m_int2enum.begin() ; ++ it ) {
    Py_CLEAR( it->second );
  }
}

void
pytools::EnumType::initType(const char* typeName)
{
  size_t len = std::strlen(typeName);
  m_typeName = new char[len + 1];
  std::strcpy(m_typeName, typeName);

  // initialize type structure
  memset ( &m_type, 0, sizeof m_type );
  memcpy( &m_type, &initHeader, sizeof initHeader );
  m_type.tp_name = m_typeName;
  m_type.tp_basicsize = sizeof(EnumObject);
  m_type.tp_dealloc = Enum_dealloc;
  m_type.tp_repr = Enum_str_int;
  m_type.tp_str = Enum_str_repr;
  m_type.tp_flags = Py_TPFLAGS_DEFAULT;
  m_type.tp_doc = 0;
  m_type.tp_base = &PyInt_Type ;
  m_type.tp_init = Enum_init ;
  m_type.tp_alloc = PyType_GenericAlloc ;
  m_type.tp_new = PyType_GenericNew ;
  m_type.tp_free = _PyObject_Del ;
  m_type.tp_del = Enum_dealloc;

  PyObject* tp_dict = PyDict_New();
  m_type.tp_dict = tp_dict;

  // finalize type
  PyType_Ready( &m_type );
}


void pytools::EnumType::makeDocString()
{
  // get sorted set of keys and values
  std::map<std::string, int> items;
  Py_ssize_t ppos = 0;
  PyObject *pkey, *pvalue;
  while (PyDict_Next(m_type.tp_dict, &ppos, &pkey, &pvalue)) {
    if (pvalue->ob_type == &m_type) {
      // pkey must be a string, what else could it be?
      items.insert(std::make_pair(std::string(PyString_AsString(pkey)), int(PyInt_AS_LONG(pvalue))));
    }
  }

  // generate doc string
  std::ostringstream doc;
  doc << ::docString << "\nThis class defines following constants:";
  for (std::map<std::string, int>::const_iterator it = items.begin(); it != items.end(); ++ it) {
    doc << " " << it->first << "=" << it->second;
  }

  const std::string& docstr = doc.str();

  if (m_docString) delete [] m_docString;
  m_docString = new char[docstr.size()+1];
  std::copy(docstr.begin(), docstr.end(), m_docString);
  m_docString[docstr.size()] = '\0';
  m_type.tp_doc = m_docString;
}

/**
 *  Add one more enum value to the type.
 */
void
pytools::EnumType::addEnum(const std::string& name, int value)
{
  // this enum class name
  const char* p = strrchr( m_type.tp_name, '.' );
  std::string ename = p ? p+1 : m_type.tp_name;
  ename += '.';

  // build the object
  EnumObject* evalue = PyObject_New(EnumObject, &m_type);
  ename += name;
  evalue->ob_ival = value;
  evalue->en_name = PyString_FromString(ename.c_str());

  // store it as the class attribute
  PyDict_SetItemString(m_type.tp_dict, name.c_str(), (PyObject*)evalue);

  // and in the map
  m_int2enum[value] = (PyObject*)evalue;

  // update doc string
  makeDocString();
}

// Make instance of this type
PyObject*
pytools::EnumType::Enum_FromLong( long value ) const
{
  Int2Enum::const_iterator it = m_int2enum.find( value );
  if ( it == m_int2enum.end() ) {
    return PyInt_FromLong(value);
  }

  Py_INCREF( it->second );
  return it->second;
}

PyObject*
pytools::EnumType::Enum_FromString( const char* name ) const
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
    pytools::EnumType* enumType = (pytools::EnumType*)self->ob_type;
    if ( PyObject* o = enumType->Enum_FromLong( PyInt_AsLong(arg) ) ) {

      if ( PyObject_TypeCheck( o, self->ob_type ) ) {
        EnumObject* enumObj = (EnumObject*)o;
        py_this->ob_ival = enumObj->ob_ival;
        py_this->en_name = enumObj->en_name;
        Py_INCREF(py_this->en_name);
      } else {
        PyIntObject* enumObj = (PyIntObject*)o;
        py_this->ob_ival = enumObj->ob_ival;
        py_this->en_name = Py_None;
        Py_INCREF(py_this->en_name);
      }

    } else {

      return -1;

    }

  } else if ( PyString_Check(arg) ) {

    // dirty hack
    pytools::EnumType* enumType = (pytools::EnumType*)self->ob_type;
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
  Py_CLEAR( ((EnumObject*)self)->en_name);

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

PyObject*
Enum_str_int( PyObject* self )
{
  long val = ((PyIntObject*) self)->ob_ival;
  char buf[64];
  snprintf(buf, sizeof buf, "%ld", val);
  return PyString_FromString(buf);
}

}
