//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Timepix_TM6740ConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV1.h"

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

  pypdsdata::EnumType::Enum speedEnumValues[] = {
      { "ReadoutSpeed_Slow",  Pds::Timepix::ConfigV1::ReadoutSpeed_Slow },
      { "ReadoutSpeed_Fast",  Pds::Timepix::ConfigV1::ReadoutSpeed_Fast },
      { 0, 0 }
  };
  pypdsdata::EnumType speedEnum ( "ReadoutSpeed", speedEnumValues );

  pypdsdata::EnumType::Enum trigEnumValues[] = {
      { "TriggerMode_ExtPos", Pds::Timepix::ConfigV1::TriggerMode_ExtPos },
      { "TriggerMode_ExtNeg", Pds::Timepix::ConfigV1::TriggerMode_ExtNeg },
      { "TriggerMode_Soft",   Pds::Timepix::ConfigV1::TriggerMode_Soft },
      { 0, 0 }
  };
  pypdsdata::EnumType trigEnum ( "TriggerMode", trigEnumValues );


  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
        { "ChipCount", Pds::Timepix::ConfigV1::ChipCount },
        { 0, 0 }
  };

  // methods
  ENUM_FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, readoutSpeed, speedEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, triggerMode, trigEnum)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, shutterTimeout)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac0Ikrum)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac0Disc)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac0Preamp)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac0BufAnalogA)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac0BufAnalogB)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac0Hist)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac0ThlFine)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac0ThlCourse)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac0Vcas)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac0Fbk)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac0Gnd)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac0Ths)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac0BiasLvds)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac0RefLvds)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac1Ikrum)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac1Disc)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac1Preamp)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac1BufAnalogA)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac1BufAnalogB)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac1Hist)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac1ThlFine)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac1ThlCourse)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac1Vcas)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac1Fbk)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac1Gnd)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac1Ths)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac1BiasLvds)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac1RefLvds)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac2Ikrum)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac2Disc)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac2Preamp)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac2BufAnalogA)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac2BufAnalogB)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac2Hist)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac2ThlFine)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac2ThlCourse)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac2Vcas)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac2Fbk)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac2Gnd)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac2Ths)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac2BiasLvds)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac2RefLvds)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac3Ikrum)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac3Disc)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac3Preamp)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac3BufAnalogA)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac3BufAnalogB)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac3Hist)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac3ThlFine)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac3ThlCourse)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac3Vcas)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac3Fbk)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac3Gnd)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac3Ths)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac3BiasLvds)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV1, dac3RefLvds)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "readoutSpeed",      readoutSpeed        , METH_NOARGS,  "self.readoutSpeed() -> ReadoutSpeed enum\n\nReturns ReadoutSpeed enum" },
    { "triggerMode",       triggerMode         , METH_NOARGS,  "self.triggerMode() -> TriggerMode enum\n\nReturns TriggerMode enum" },
    { "shutterTimeout",    shutterTimeout      , METH_NOARGS,  "self.shutterTimeout() -> int\n\nReturns integer number" },
    { "dac0Ikrum",         dac0Ikrum           , METH_NOARGS,  "self.dac0Ikrum() -> int\n\nReturns integer number" },
    { "dac0Disc",          dac0Disc            , METH_NOARGS,  "self.dac0Disc() -> int\n\nReturns integer number" },
    { "dac0Preamp",        dac0Preamp          , METH_NOARGS,  "self.dac0Preamp() -> int\n\nReturns integer number" },
    { "dac0BufAnalogA",    dac0BufAnalogA      , METH_NOARGS,  "self.dac0BufAnalogA() -> int\n\nReturns integer number" },
    { "dac0BufAnalogB",    dac0BufAnalogB      , METH_NOARGS,  "self.dac0BufAnalogB() -> int\n\nReturns integer number" },
    { "dac0Hist",          dac0Hist            , METH_NOARGS,  "self.dac0Hist() -> int\n\nReturns integer number" },
    { "dac0ThlFine",       dac0ThlFine         , METH_NOARGS,  "self.dac0ThlFine() -> int\n\nReturns integer number" },
    { "dac0ThlCourse",     dac0ThlCourse       , METH_NOARGS,  "self.dac0ThlCourse() -> int\n\nReturns integer number" },
    { "dac0Vcas",          dac0Vcas            , METH_NOARGS,  "self.dac0Vcas() -> int\n\nReturns integer number" },
    { "dac0Fbk",           dac0Fbk             , METH_NOARGS,  "self.dac0Fbk() -> int\n\nReturns integer number" },
    { "dac0Gnd",           dac0Gnd             , METH_NOARGS,  "self.dac0Gnd() -> int\n\nReturns integer number" },
    { "dac0Ths",           dac0Ths             , METH_NOARGS,  "self.dac0Ths() -> int\n\nReturns integer number" },
    { "dac0BiasLvds",      dac0BiasLvds        , METH_NOARGS,  "self.dac0BiasLvds() -> int\n\nReturns integer number" },
    { "dac0RefLvds",       dac0RefLvds         , METH_NOARGS,  "self.dac0RefLvds() -> int\n\nReturns integer number" },
    { "dac1Ikrum",         dac1Ikrum           , METH_NOARGS,  "self.dac1Ikrum() -> int\n\nReturns integer number" },
    { "dac1Disc",          dac1Disc            , METH_NOARGS,  "self.dac1Disc() -> int\n\nReturns integer number" },
    { "dac1Preamp",        dac1Preamp          , METH_NOARGS,  "self.dac1Preamp() -> int\n\nReturns integer number" },
    { "dac1BufAnalogA",    dac1BufAnalogA      , METH_NOARGS,  "self.dac1BufAnalogA() -> int\n\nReturns integer number" },
    { "dac1BufAnalogB",    dac1BufAnalogB      , METH_NOARGS,  "self.dac1BufAnalogB() -> int\n\nReturns integer number" },
    { "dac1Hist",          dac1Hist            , METH_NOARGS,  "self.dac1Hist() -> int\n\nReturns integer number" },
    { "dac1ThlFine",       dac1ThlFine         , METH_NOARGS,  "self.dac1ThlFine() -> int\n\nReturns integer number" },
    { "dac1ThlCourse",     dac1ThlCourse       , METH_NOARGS,  "self.dac1ThlCourse() -> int\n\nReturns integer number" },
    { "dac1Vcas",          dac1Vcas            , METH_NOARGS,  "self.dac1Vcas() -> int\n\nReturns integer number" },
    { "dac1Fbk",           dac1Fbk             , METH_NOARGS,  "self.dac1Fbk() -> int\n\nReturns integer number" },
    { "dac1Gnd",           dac1Gnd             , METH_NOARGS,  "self.dac1Gnd() -> int\n\nReturns integer number" },
    { "dac1Ths",           dac1Ths             , METH_NOARGS,  "self.dac1Ths() -> int\n\nReturns integer number" },
    { "dac1BiasLvds",      dac1BiasLvds        , METH_NOARGS,  "self.dac1BiasLvds() -> int\n\nReturns integer number" },
    { "dac1RefLvds",       dac1RefLvds         , METH_NOARGS,  "self.dac1RefLvds() -> int\n\nReturns integer number" },
    { "dac2Ikrum",         dac2Ikrum           , METH_NOARGS,  "self.dac2Ikrum() -> int\n\nReturns integer number" },
    { "dac2Disc",          dac2Disc            , METH_NOARGS,  "self.dac2Disc() -> int\n\nReturns integer number" },
    { "dac2Preamp",        dac2Preamp          , METH_NOARGS,  "self.dac2Preamp() -> int\n\nReturns integer number" },
    { "dac2BufAnalogA",    dac2BufAnalogA      , METH_NOARGS,  "self.dac2BufAnalogA() -> int\n\nReturns integer number" },
    { "dac2BufAnalogB",    dac2BufAnalogB      , METH_NOARGS,  "self.dac2BufAnalogB() -> int\n\nReturns integer number" },
    { "dac2Hist",          dac2Hist            , METH_NOARGS,  "self.dac2Hist() -> int\n\nReturns integer number" },
    { "dac2ThlFine",       dac2ThlFine         , METH_NOARGS,  "self.dac2ThlFine() -> int\n\nReturns integer number" },
    { "dac2ThlCourse",     dac2ThlCourse       , METH_NOARGS,  "self.dac2ThlCourse() -> int\n\nReturns integer number" },
    { "dac2Vcas",          dac2Vcas            , METH_NOARGS,  "self.dac2Vcas() -> int\n\nReturns integer number" },
    { "dac2Fbk",           dac2Fbk             , METH_NOARGS,  "self.dac2Fbk() -> int\n\nReturns integer number" },
    { "dac2Gnd",           dac2Gnd             , METH_NOARGS,  "self.dac2Gnd() -> int\n\nReturns integer number" },
    { "dac2Ths",           dac2Ths             , METH_NOARGS,  "self.dac2Ths() -> int\n\nReturns integer number" },
    { "dac2BiasLvds",      dac2BiasLvds        , METH_NOARGS,  "self.dac2BiasLvds() -> int\n\nReturns integer number" },
    { "dac2RefLvds",       dac2RefLvds         , METH_NOARGS,  "self.dac2RefLvds() -> int\n\nReturns integer number" },
    { "dac3Ikrum",         dac3Ikrum           , METH_NOARGS,  "self.dac3Ikrum() -> int\n\nReturns integer number" },
    { "dac3Disc",          dac3Disc            , METH_NOARGS,  "self.dac3Disc() -> int\n\nReturns integer number" },
    { "dac3Preamp",        dac3Preamp          , METH_NOARGS,  "self.dac3Preamp() -> int\n\nReturns integer number" },
    { "dac3BufAnalogA",    dac3BufAnalogA      , METH_NOARGS,  "self.dac3BufAnalogA() -> int\n\nReturns integer number" },
    { "dac3BufAnalogB",    dac3BufAnalogB      , METH_NOARGS,  "self.dac3BufAnalogB() -> int\n\nReturns integer number" },
    { "dac3Hist",          dac3Hist            , METH_NOARGS,  "self.dac3Hist() -> int\n\nReturns integer number" },
    { "dac3ThlFine",       dac3ThlFine         , METH_NOARGS,  "self.dac3ThlFine() -> int\n\nReturns integer number" },
    { "dac3ThlCourse",     dac3ThlCourse       , METH_NOARGS,  "self.dac3ThlCourse() -> int\n\nReturns integer number" },
    { "dac3Vcas",          dac3Vcas            , METH_NOARGS,  "self.dac3Vcas() -> int\n\nReturns integer number" },
    { "dac3Fbk",           dac3Fbk             , METH_NOARGS,  "self.dac3Fbk() -> int\n\nReturns integer number" },
    { "dac3Gnd",           dac3Gnd             , METH_NOARGS,  "self.dac3Gnd() -> int\n\nReturns integer number" },
    { "dac3Ths",           dac3Ths             , METH_NOARGS,  "self.dac3Ths() -> int\n\nReturns integer number" },
    { "dac3BiasLvds",      dac3BiasLvds        , METH_NOARGS,  "self.dac3BiasLvds() -> int\n\nReturns integer number" },
    { "dac3RefLvds",       dac3RefLvds         , METH_NOARGS,  "self.dac3RefLvds() -> int\n\nReturns integer number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Timepix::ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Timepix::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "ReadoutSpeed", speedEnum.type() );
  PyDict_SetItemString( tp_dict, "TriggerMode", trigEnum.type() );
  pypdsdata::TypeLib::DefineEnums( tp_dict, ::enums );
  type->tp_dict = tp_dict;

  BaseType::initType( "ConfigV1", module );
}

namespace {
  
PyObject*
_repr( PyObject *self )
{
  Pds::Timepix::ConfigV1* obj = pypdsdata::Timepix::ConfigV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "Timepix.ConfigV1(readoutSpeed=" << int(obj->readoutSpeed())
      << ", triggerMode=" << int(obj->triggerMode())
      << ", shutterTimeout=" << obj->shutterTimeout()
      << ", ...)" ;

  return PyString_FromString( str.str().c_str() );
}

}
