//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_OutputMapV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "OutputMapV2.h"

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
      { "Pulse",      Pds::EvrData::OutputMapV2::Pulse },
      { "DBus",       Pds::EvrData::OutputMapV2::DBus },
      { "Prescaler",  Pds::EvrData::OutputMapV2::Prescaler },
      { "Force_High", Pds::EvrData::OutputMapV2::Force_High },
      { "Force_Low",  Pds::EvrData::OutputMapV2::Force_Low },
      { 0, 0 }
  };
  pypdsdata::EnumType sourceEnum ( "Source", sourceEnumValues );

  pypdsdata::EnumType::Enum connEnumValues[] = {
      { "FrontPanel", Pds::EvrData::OutputMapV2::FrontPanel },
      { "UnivIO",     Pds::EvrData::OutputMapV2::UnivIO },
      { 0, 0 }
  };
  pypdsdata::EnumType connEnum ( "Conn", connEnumValues );

  // type-specific methods
  ENUM_FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::OutputMapV2, source, sourceEnum)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::OutputMapV2, source_id)
  ENUM_FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::OutputMapV2, conn, connEnum)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::OutputMapV2, conn_id)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::OutputMapV2, module)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::OutputMapV2, map)

  PyMethodDef methods[] = {
    { "source",    source,     METH_NOARGS, "self.source() -> Source enum\n\nReturns source (generated pulse) of output generation (:py:class:`Source`)" },
    { "source_id", source_id,  METH_NOARGS, "self.source_id() -> int\n\nReturns source (generated pulse) of output generation" },
    { "conn",      conn,       METH_NOARGS, "self.conn() -> Conn enum\n\nReturns connector for output destination (:py:class:`Conn`)" },
    { "conn_id",   conn_id,    METH_NOARGS, "self.conn_id() -> int\n\nReturns connector for output destination" },
    { "module",    module,     METH_NOARGS, "self.module() -> int\n\nReturns module number" },
    { "map",       map,        METH_NOARGS, "self.map() -> int\n\nReturns encoded source value" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::OutputMapV2 class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::OutputMapV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Source", ::sourceEnum.type() );
  PyDict_SetItemString( tp_dict, "Conn", ::connEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "OutputMapV2", module );
}

pypdsdata::EnumType& 
pypdsdata::EvrData::OutputMapV2::connEnum() 
{
  return ::connEnum;
}
