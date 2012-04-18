//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcVetoIO...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TdcVetoIO.h"

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
      { "ChVeto",    Pds::Acqiris::TdcVetoIO::ChVeto },
      { 0, 0 }
  };
  pypdsdata::EnumType chanEnum ( "Channel", chanEnumValues );

  pypdsdata::EnumType::Enum modeEnumValues[] = {
      { "Veto",               Pds::Acqiris::TdcVetoIO::Veto },
      { "SwitchVeto",         Pds::Acqiris::TdcVetoIO::SwitchVeto },
      { "InvertedVeto",       Pds::Acqiris::TdcVetoIO::InvertedVeto },
      { "InvertedSwitchVeto", Pds::Acqiris::TdcVetoIO::InvertedSwitchVeto },
      { 0, 0 }
  };
  pypdsdata::EnumType modeEnum ( "Mode", modeEnumValues );

  pypdsdata::EnumType::Enum termEnumValues[] = {
      { "ZHigh",  Pds::Acqiris::TdcVetoIO::ZHigh },
      { "Z50",    Pds::Acqiris::TdcVetoIO::Z50 },
      { 0, 0 }
  };
  pypdsdata::EnumType termEnum ( "Termination", termEnumValues );

  // methods
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TdcVetoIO, channel, chanEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TdcVetoIO, mode, modeEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TdcVetoIO, term, termEnum)

  PyMethodDef methods[] = {
    {"channel",   channel,  METH_NOARGS,  "self.channel() -> Channel enum\n\nReturns :py:class:`TdcVetoIO.Channel` enum value" },
    {"mode",      mode,     METH_NOARGS,  "self.mode() -> Mode enum\n\nReturns :py:class:`TdcVetoIO.Mode` enum value" },
    {"term",      term,     METH_NOARGS,  "self.term() -> Termination enum\n\nReturns :py:class:`TdcVetoIO.Termination` enum value" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::TdcVetoIO class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::TdcVetoIO::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums and constants
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Channel", ::chanEnum.type() );
  PyDict_SetItemString( tp_dict, "Mode", ::modeEnum.type() );
  PyDict_SetItemString( tp_dict, "Termination", ::termEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "TdcVetoIO", module );
}

void 
pypdsdata::Acqiris::TdcVetoIO::print(std::ostream& out) const
{
  if(not m_obj) {
    out << "acqiris.TdcVetoIO(None)";
  } else {  
    out << "acqiris.TdcVetoIO(channel=" << m_obj->channel()
        << ", mode=" << m_obj->mode()
        << ", term=" << m_obj->term()
        << ")" ;
  }
}
