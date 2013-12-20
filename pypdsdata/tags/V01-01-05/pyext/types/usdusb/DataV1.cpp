//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class UsdUsb_DataV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataV1.h"

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

  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
        { "Encoder_Inputs", Pds::UsdUsb::DataV1::Encoder_Inputs },
        { "Analog_Inputs", Pds::UsdUsb::DataV1::Analog_Inputs },
        { "Digital_Inputs", Pds::UsdUsb::DataV1::Digital_Inputs },
        { 0, 0 }
  };

  // methods
  FUN0_WRAPPER(pypdsdata::UsdUsb::DataV1, digital_in)
  FUN0_WRAPPER(pypdsdata::UsdUsb::DataV1, timestamp)
  FUN0_WRAPPER(pypdsdata::UsdUsb::DataV1, status)
  FUN0_WRAPPER(pypdsdata::UsdUsb::DataV1, encoder_count)
  FUN0_WRAPPER(pypdsdata::UsdUsb::DataV1, analog_in)

  PyMethodDef methods[] = {
    {"digital_in",     digital_in,      METH_NOARGS,  "self.digital_in() -> int\n\nReturns integer number" },
    {"timestamp",      timestamp,       METH_NOARGS,  "self.timestamp() -> int\n\nReturns integer number" },
    {"status",         status,          METH_NOARGS,  "self.status() -> list of ints\n\nReturns list of integers" },
    {"encoder_count",  encoder_count,   METH_NOARGS,  "self.encoder_count() -> list of ints\n\nReturns list of Encoder_Inputs numbers" },
    {"analog_in",      analog_in,       METH_NOARGS,  "self.analog_in() -> list of ints\n\nReturns list of Analog_Inputs numbers" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::UsdUsb::DataV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::UsdUsb::DataV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  pypdsdata::TypeLib::DefineEnums( tp_dict, ::enums );
  type->tp_dict = tp_dict;

  BaseType::initType( "DataV1", module );
}

void
pypdsdata::UsdUsb::DataV1::print(std::ostream& str) const
{
  str << "usdusb.DataV1(encoder_count=" << m_obj->encoder_count()
      << ", analog_in=" << m_obj->analog_in()
      << ", digital_in=" << m_obj->digital_in()
      << ", timestamp=" << m_obj->timestamp()
      << ", status=" << m_obj->status()
      << ")" ;
}
