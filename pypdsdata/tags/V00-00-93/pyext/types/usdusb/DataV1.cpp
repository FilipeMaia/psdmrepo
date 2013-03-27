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
  PyObject* encoder_count( PyObject *self, PyObject* );
  PyObject* analog_in( PyObject *self, PyObject* );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"encoder_count",  encoder_count,   METH_NOARGS,  "self.encoder_count() -> list of ints\n\nReturns list of Encoder_Inputs numbers" },
    {"analog_in",      analog_in,       METH_NOARGS,  "self.analog_in() -> list of ints\n\nReturns list of Analog_Inputs numbers" },
    {"digital_in",     digital_in,      METH_NOARGS,  "self.digital_in() -> int\n\nReturns integer number" },
    {"timestamp",      timestamp,       METH_NOARGS,  "self.timestamp() -> int\n\nReturns integer number" },
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
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  pypdsdata::TypeLib::DefineEnums( tp_dict, ::enums );
  type->tp_dict = tp_dict;

  BaseType::initType( "DataV1", module );
}

namespace {

PyObject*
encoder_count( PyObject* self, PyObject* )
{
  Pds::UsdUsb::DataV1* obj = pypdsdata::UsdUsb::DataV1::pdsObject(self);
  if(not obj) return 0;

  PyObject* list = PyList_New(Pds::UsdUsb::DataV1::Encoder_Inputs);

  // copy coordinates to the list
  for ( unsigned i = 0; i < Pds::UsdUsb::DataV1::Encoder_Inputs; ++ i ) {
    PyList_SET_ITEM( list, i, PyInt_FromLong(obj->encoder_count(i)) );
  }

  return list;
}

PyObject*
analog_in( PyObject* self, PyObject* )
{
  Pds::UsdUsb::DataV1* obj = pypdsdata::UsdUsb::DataV1::pdsObject(self);
  if(not obj) return 0;

  PyObject* list = PyList_New(Pds::UsdUsb::DataV1::Analog_Inputs);

  // copy coordinates to the list
  for ( unsigned i = 0; i < Pds::UsdUsb::DataV1::Analog_Inputs; ++ i ) {
    PyList_SET_ITEM( list, i, PyInt_FromLong(obj->analog_in(i)) );
  }

  return list;
}

PyObject*
_repr( PyObject *self )
{
  Pds::UsdUsb::DataV1* obj = pypdsdata::UsdUsb::DataV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "usdusb.DataV1(encoder_count=[";
  for (int i = 0; i != Pds::UsdUsb::DataV1::Encoder_Inputs; ++ i) {
    if (i != 0) str << ", ";
    str << obj->encoder_count(i);
  }
  str << "], analog_in=[";
  for (int i = 0; i != Pds::UsdUsb::DataV1::Analog_Inputs; ++ i) {
    if (i != 0) str << ", ";
    str << obj->analog_in(i);
  }
  str << "], digital_in=" << obj->digital_in()
      << ", timestamp=" << obj->timestamp()
      << ")" ;

  return PyString_FromString( str.str().c_str() );
}

}
