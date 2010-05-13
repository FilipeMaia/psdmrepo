//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_OutputMap...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "OutputMap.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <new>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../EnumType.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  pypdsdata::EnumType::Enum sourceEnumValues[] = {
      { "Pulse",      Pds::EvrData::OutputMap::Pulse },
      { "DBus",       Pds::EvrData::OutputMap::DBus },
      { "Prescaler",  Pds::EvrData::OutputMap::Prescaler },
      { "Force_High", Pds::EvrData::OutputMap::Force_High },
      { "Force_Low",  Pds::EvrData::OutputMap::Force_Low },
      { 0, 0 }
  };
  pypdsdata::EnumType sourceEnum ( "Source", sourceEnumValues );

  pypdsdata::EnumType::Enum connEnumValues[] = {
      { "FrontPanel", Pds::EvrData::OutputMap::FrontPanel },
      { "UnivIO",     Pds::EvrData::OutputMap::UnivIO },
      { 0, 0 }
  };
  pypdsdata::EnumType connEnum ( "Conn", connEnumValues );

  // type-specific methods
  ENUM_FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::OutputMap, source, sourceEnum)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::OutputMap, source_id)
  ENUM_FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::OutputMap, conn, connEnum)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::OutputMap, conn_id)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::OutputMap, map)

  PyMethodDef methods[] = {
    { "source",    source,     METH_NOARGS, "source (generated pulse) of output generation" },
    { "source_id", source_id,  METH_NOARGS, "source (generated pulse) of output generation" },
    { "conn",      conn,       METH_NOARGS, "connector for output destination" },
    { "conn_id",   conn_id,    METH_NOARGS, "connector for output destination" },
    { "map",       map,        METH_NOARGS, "encoded source value" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::OutputMap class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::OutputMap::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Source", sourceEnum.type() );
  PyDict_SetItemString( tp_dict, "Conn", connEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "OutputMap", module );
}
