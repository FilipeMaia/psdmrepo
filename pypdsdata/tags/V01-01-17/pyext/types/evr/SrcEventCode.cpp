//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_SrcEventCode...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "SrcEventCode.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iomanip>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::SrcEventCode, code)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::SrcEventCode, period)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::SrcEventCode, maskTriggerP)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::SrcEventCode, maskTriggerR)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::SrcEventCode, desc)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::SrcEventCode, readoutGroup)

  PyMethodDef methods[] = {
    { "code",          code,         METH_NOARGS, "self.code() -> int\n\nReturns assigned eventcode." },
    { "period",        period,       METH_NOARGS, "self.period() -> int\n\nReturns repetition period in 119 MHz counts or 0 for external source." },
    { "maskTriggerP",  maskTriggerP, METH_NOARGS, "self.maskTriggerP() -> int\n\nReturns bit mask of persistent pulse triggers." },
    { "maskTriggerR",  maskTriggerR, METH_NOARGS, "self.maskTriggerR() -> int\n\nReturns bit mask of running pulse triggers." },
    { "desc",          desc,         METH_NOARGS, "self.desc() -> string\n\nReturns optional description." },
    { "readoutGroup",  readoutGroup, METH_NOARGS, "self.readoutGroup() -> int\n\nReturns assigned readout group" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::SrcEventCode class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::SrcEventCode::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyObject* val = PyInt_FromLong(Pds::EvrData::SrcEventCode::MaxReadoutGroup);
  PyDict_SetItemString( type->tp_dict, "MaxReadoutGroup", val );
  Py_XDECREF(val);

  BaseType::initType( "SrcEventCode", module );
}

void
pypdsdata::EvrData::SrcEventCode::print(std::ostream& str) const
{
  str << "EvrData.SrcEventCode(code=" << m_obj.code()
      << ", period=" << m_obj.period()
      << std::hex
      << ", maskTriggerP=" << m_obj.maskTriggerP()
      << ", maskTriggerR=" << m_obj.maskTriggerR()
      << std::dec
      << ", readoutGroup=" << m_obj.readoutGroup()
      << ", desc=\"" << m_obj.desc() << "\""
      << ")" ;
}
