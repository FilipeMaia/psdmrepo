//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcAuxIO...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TdcAuxIO.h"

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
      { "IOAux1",    Pds::Acqiris::TdcAuxIO::IOAux1 },
      { "IOAux2",    Pds::Acqiris::TdcAuxIO::IOAux2 },
      { 0, 0 }
  };
  pypdsdata::EnumType chanEnum ( "Channel", chanEnumValues );

  pypdsdata::EnumType::Enum modeEnumValues[] = {
      { "BankSwitch", Pds::Acqiris::TdcAuxIO::BankSwitch },
      { "Marker",     Pds::Acqiris::TdcAuxIO::Marker },
      { "OutputLo",   Pds::Acqiris::TdcAuxIO::OutputLo },
      { "OutputHi",   Pds::Acqiris::TdcAuxIO::OutputHi },
      { 0, 0 }
  };
  pypdsdata::EnumType modeEnum ( "Mode", modeEnumValues );

  pypdsdata::EnumType::Enum termEnumValues[] = {
      { "ZHigh",  Pds::Acqiris::TdcAuxIO::ZHigh },
      { "Z50",    Pds::Acqiris::TdcAuxIO::Z50 },
      { 0, 0 }
  };
  pypdsdata::EnumType termEnum ( "Termination", termEnumValues );

  // methods
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TdcAuxIO, channel, chanEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TdcAuxIO, mode, modeEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TdcAuxIO, term, termEnum)

  PyMethodDef methods[] = {
    {"channel",   channel,  METH_NOARGS,  "self.channel() -> Channel enum\n\nReturns Channel enum value" },
    {"mode",      mode,     METH_NOARGS,  "self.mode() -> Mode enum\n\nReturns Mode enum value" },
    {"term",      term,     METH_NOARGS,  "self.term() -> Termination enum\n\nReturns Termination enum value" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::TdcAuxIO class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::TdcAuxIO::initType( PyObject* module )
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

  BaseType::initType( "TdcAuxIO", module );
}

void 
pypdsdata::Acqiris::TdcAuxIO::print(std::ostream& out) const
{
  if(not m_obj) {
    out << "acqiris.TdcAuxIO(None)";
  } else {  
    out << "acqiris.TdcAuxIO(channel=" << m_obj->channel()
        << ", mode=" << m_obj->mode()
        << ", term=" << m_obj->term()
        << ")" ;
  }
}
