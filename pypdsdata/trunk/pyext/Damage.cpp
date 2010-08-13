//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Damage...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Damage.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <new>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "EnumType.h"
#include "types/TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  pypdsdata::EnumType::Enum valueEnumValues[] = {
      { "DroppedContribution",    Pds::Damage::DroppedContribution },
      { "OutOfOrder",             Pds::Damage::OutOfOrder },
      { "OutOfSynch",             Pds::Damage::OutOfSynch },
      { "UserDefined",             Pds::Damage::UserDefined },
      { "IncompleteContribution", Pds::Damage::IncompleteContribution },
      { "ContainsIncomplete",     Pds::Damage::ContainsIncomplete },
      { 0, 0 }
  };
  pypdsdata::EnumType valueEnum ( "Value", valueEnumValues );

  pypdsdata::EnumType::Enum maskEnumValues[] = {
      { "DroppedContribution",    1 << Pds::Damage::DroppedContribution },
      { "OutOfOrder",             1 << Pds::Damage::OutOfOrder },
      { "OutOfSynch",             1 << Pds::Damage::OutOfSynch },
      { "UserDefined",             1 << Pds::Damage::UserDefined },
      { "IncompleteContribution", 1 << Pds::Damage::IncompleteContribution },
      { "ContainsIncomplete",     1 << Pds::Damage::ContainsIncomplete },
      { 0, 0 }
  };
  pypdsdata::EnumType maskEnum ( "Mask", maskEnumValues );

  // standard Python stuff
  int Damage_init( PyObject* self, PyObject* args, PyObject* kwds );
  PyObject* Damage_str( PyObject* self );
  PyObject* Damage_repr( PyObject* self );

  // type-specific methods
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Damage, value);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Damage, bits);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Damage, userBits);
  PyObject* Damage_hasDamage( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    { "value",     value,            METH_NOARGS,  "Returns complete damage mask" },
    { "bits",      bits,             METH_NOARGS,  "Returns damage mask excluding user bits" },
    { "userBits",  userBits,         METH_NOARGS,  "Returns user bits of the damage mask" },
    { "hasDamage", Damage_hasDamage, METH_VARARGS, "Returns True if the damage bit is set, accepts values like Damage.OutOfOrder" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Damage class.\n\n"
      "Class constructor takes zero or one integer numbers, constructor with zero\n"
      "arguments will create no-damage object.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Damage::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_init = Damage_init;
  type->tp_str = Damage_str;
  type->tp_repr = Damage_repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Value", valueEnum.type() );
  PyDict_SetItemString( tp_dict, "Mask", maskEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "Damage", module );
}

namespace {

int
Damage_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::Damage* py_this = (pypdsdata::Damage*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  unsigned val = 0;
  if ( not PyArg_ParseTuple( args, "|I:Damage", &val ) ) return -1;

  new(&py_this->m_obj) Pds::Damage(val);

  return 0;
}

PyObject*
Damage_str( PyObject* self )
{
  return Damage_repr(self);
}

PyObject*
Damage_repr( PyObject* self )
{
  pypdsdata::Damage* py_this = (pypdsdata::Damage*) self;
  return PyString_FromFormat("<Damage(%d)>", int(py_this->m_obj.value()) );
}

PyObject*
Damage_hasDamage( PyObject* self, PyObject* args )
{
  pypdsdata::Damage* py_this = (pypdsdata::Damage*) self;

  unsigned bit;
  if ( not PyArg_ParseTuple( args, "I:damage.hasDamage", &bit ) ) return 0;

  return PyBool_FromLong( py_this->m_obj.value() & (1 << bit) );
}

}
