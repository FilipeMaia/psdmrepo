//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcChannel...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TdcChannel.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  pypdsdata::EnumType::Enum chanEnumValues[] = {
      { "Veto",    Pds::Acqiris::TdcChannel::Veto },
      { "Common",  Pds::Acqiris::TdcChannel::Common },
      { "Input1",  Pds::Acqiris::TdcChannel::Input1 },
      { "Input2",  Pds::Acqiris::TdcChannel::Input2 },
      { "Input3",  Pds::Acqiris::TdcChannel::Input3 },
      { "Input4",  Pds::Acqiris::TdcChannel::Input4 },
      { "Input5",  Pds::Acqiris::TdcChannel::Input5 },
      { "Input6",  Pds::Acqiris::TdcChannel::Input6 },
      { 0, 0 }
  };
  pypdsdata::EnumType chanEnum ( "Channel", chanEnumValues );

  pypdsdata::EnumType::Enum modeEnumValues[] = {
      { "Active",    Pds::Acqiris::TdcChannel::Active },
      { "Inactive",  Pds::Acqiris::TdcChannel::Inactive },
      { 0, 0 }
  };
  pypdsdata::EnumType modeEnum ( "Mode", modeEnumValues );

  pypdsdata::EnumType::Enum slopeEnumValues[] = {
      { "Positive",  Pds::Acqiris::TdcChannel::Positive },
      { "Negative",  Pds::Acqiris::TdcChannel::Negative },
      { 0, 0 }
  };
  pypdsdata::EnumType slopeEnum ( "Slope", slopeEnumValues );

  // methods
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TdcChannel, channel, chanEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TdcChannel, mode, modeEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TdcChannel, slope, slopeEnum)
  FUN0_WRAPPER(pypdsdata::Acqiris::TdcChannel, level)

  PyMethodDef methods[] = {
    {"channel",   channel,  METH_NOARGS,  "self.channel() -> Channel enum\n\nReturns Channel enum value" },
    {"mode",      mode,     METH_NOARGS,  "self.mode() -> Mode enum\n\nReturns Mode enum value" },
    {"slope",     slope,    METH_NOARGS,  "self.slope() -> Slope enum\n\nReturns Slope enum value" },
    {"level",     level,    METH_NOARGS,  "self.level() -> float\n\nReturns floating number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::TdcChannel class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::TdcChannel::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums and constants
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Channel", ::chanEnum.type() );
  PyDict_SetItemString( tp_dict, "Mode", ::modeEnum.type() );
  PyDict_SetItemString( tp_dict, "Slope", ::slopeEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "TdcChannel", module );
}

void 
pypdsdata::Acqiris::TdcChannel::print(std::ostream& out) const
{
  if(not m_obj) {
    out << "acqiris.TdcChannel(None)";
  } else {  
    out << "acqiris.TdcChannel(channel=" << m_obj->channel()
        << ", mode=" << m_obj->mode()
        << ", slope=" << m_obj->slope()
        << ", level=" << m_obj->level()
        << ")" ;
  }
}
