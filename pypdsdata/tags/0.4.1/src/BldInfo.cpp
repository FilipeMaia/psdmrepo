//--------------------------------------------------------------------------
// File and Version Information:
//  $Id$
//
// Description:
//  Class BldInfo...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldInfo.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <new>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "EnumType.h"
#include "Level.h"
#include "types/TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

    pypdsdata::EnumType::Enum typeEnumValues[] = {
        { "EBeam",           Pds::BldInfo::EBeam },
        { "PhaseCavity",     Pds::BldInfo::PhaseCavity },
        { "FEEGasDetEnergy", Pds::BldInfo::FEEGasDetEnergy },
        { "NumberOf",        Pds::BldInfo::NumberOf },
        { 0, 0 }
    };
    pypdsdata::EnumType typeEnum ( "Type", typeEnumValues );


  // standard Python stuff
  int BldInfo_init(PyObject* self, PyObject* args, PyObject* kwds);
  long BldInfo_hash( PyObject* self );
  int BldInfo_compare( PyObject *self, PyObject *other);
  PyObject* BldInfo_str( PyObject *self );
  PyObject* BldInfo_repr( PyObject *self );

  // type-specific methods
  PyObject* BldInfo_level( PyObject* self, PyObject* args );
  FUN0_WRAPPER_EMBEDDED(pypdsdata::BldInfo, log);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::BldInfo, phy);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::BldInfo, processId);
  ENUM_FUN0_WRAPPER_EMBEDDED(pypdsdata::BldInfo, type, typeEnum);

  PyMethodDef methods[] = {
    { "level",     BldInfo_level,     METH_NOARGS, "Returns source level object (Level class)" },
    { "log",       log,       METH_NOARGS, "Returns logical address of data source" },
    { "phy",       phy,       METH_NOARGS, "Returns physical address of data source" },
    { "processId", processId, METH_NOARGS, "Returns process ID" },
    { "type",      type,      METH_NOARGS, "Returns BldInfo type" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::BldInfo class.\n\n"
      "Constructor takes two positional arguments, same values as the\n"
      "C++ constructor. Class implements usual comparison operators\n"
      "and hash function so that objects can be used as dictionary keys.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::BldInfo::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_init = BldInfo_init;
  type->tp_hash = BldInfo_hash;
  type->tp_compare = BldInfo_compare;
  type->tp_str = BldInfo_str;
  type->tp_repr = BldInfo_repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Type", typeEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "BldInfo", module );
}

namespace {

int
BldInfo_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  unsigned processId, type;
  if ( not PyArg_ParseTuple( args, "II:BldInfo", &processId, &type ) ) return -1;

  if ( type >= Pds::BldInfo::NumberOf ) {
    PyErr_SetString(PyExc_TypeError, "Error: type out of range");
    return -1;
  }

  new(&py_this->m_obj) Pds::BldInfo( processId, Pds::BldInfo::Type(type) );

  return 0;
}

long
BldInfo_hash( PyObject* self )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  int64_t log = py_this->m_obj.log() ;
  int64_t phy = py_this->m_obj.phy() ;
  long hash = log | ( phy << 32 ) ;
  return hash;
}

int
BldInfo_compare( PyObject* self, PyObject* other )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  pypdsdata::BldInfo* py_other = (pypdsdata::BldInfo*) other;
  if ( py_this->m_obj.log() > py_other->m_obj.log() ) return 1 ;
  if ( py_this->m_obj.log() < py_other->m_obj.log() ) return -1 ;
  if ( py_this->m_obj.phy() > py_other->m_obj.phy() ) return 1 ;
  if ( py_this->m_obj.phy() < py_other->m_obj.phy() ) return -1 ;
  return 0 ;
}

PyObject*
BldInfo_str( PyObject *self )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  return PyString_FromString( Pds::BldInfo::name(py_this->m_obj) );
}

PyObject*
BldInfo_repr( PyObject *self )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  char buf[32];
  snprintf( buf, sizeof buf, "<BldInfo(%d, %s)>",
      py_this->m_obj.processId(),
      Pds::BldInfo::name(py_this->m_obj) );
  return PyString_FromString( buf );
}

PyObject*
BldInfo_level( PyObject* self, PyObject* )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  return pypdsdata::Level::Level_FromInt( py_this->m_obj.level() );
}

}
