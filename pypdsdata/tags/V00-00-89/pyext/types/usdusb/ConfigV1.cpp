//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class UsdUsb_ConfigV1...
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

  pypdsdata::EnumType::Enum countModeEnumValues[] = {
      { "WRAP_FULL",   Pds::UsdUsb::ConfigV1::WRAP_FULL },
      { "LIMIT",       Pds::UsdUsb::ConfigV1::LIMIT },
      { "HALT",        Pds::UsdUsb::ConfigV1::HALT },
      { "WRAP_PRESET", Pds::UsdUsb::ConfigV1::WRAP_PRESET },
      { 0, 0 }
  };
  pypdsdata::EnumType countModeEnum ( "Count_Mode", countModeEnumValues );

  pypdsdata::EnumType::Enum quadModeEnumValues[] = {
      { "CLOCK_DIR",   Pds::UsdUsb::ConfigV1::CLOCK_DIR },
      { "X1",          Pds::UsdUsb::ConfigV1::X1 },
      { "X2",          Pds::UsdUsb::ConfigV1::X2 },
      { "X4",          Pds::UsdUsb::ConfigV1::X4 },
      { 0, 0 }
  };
  pypdsdata::EnumType quadModeEnum ( "Quad_Mode", quadModeEnumValues );

  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
        { "NCHANNELS", Pds::UsdUsb::ConfigV1::NCHANNELS },
        { 0, 0 }
  };

  // methods
  PyObject* counting_mode( PyObject *self, PyObject* );
  PyObject* quadrature_mode( PyObject *self, PyObject* );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"counting_mode",   counting_mode,   METH_NOARGS,
        "self.counting_mode() -> list of ints\n\nReturns list of NCHANNEL numbers corresponding to Count_Mode values" },
    {"quadrature_mode", quadrature_mode, METH_NOARGS,
        "self.quadrature_mode() -> list of ints\n\nReturns list of NCHANNEL numbers corresponding to Quad_Mode values" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::UsdUsb::ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::UsdUsb::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Count_Mode", countModeEnum.type() );
  PyDict_SetItemString( tp_dict, "Quad_Mode", quadModeEnum.type() );
  pypdsdata::TypeLib::DefineEnums( tp_dict, ::enums );
  type->tp_dict = tp_dict;

  BaseType::initType( "ConfigV1", module );
}

namespace {

PyObject*
counting_mode( PyObject* self, PyObject* )
{
  Pds::UsdUsb::ConfigV1* obj = pypdsdata::UsdUsb::ConfigV1::pdsObject(self);
  if(not obj) return 0;

  PyObject* list = PyList_New(Pds::UsdUsb::ConfigV1::NCHANNELS);

  // copy coordinates to the list
  for ( unsigned i = 0; i < Pds::UsdUsb::ConfigV1::NCHANNELS; ++ i ) {
    PyObject* eobj = countModeEnum.Enum_FromLong(obj->counting_mode(i));
    PyList_SET_ITEM( list, i, eobj );
  }

  return list;
}

PyObject*
quadrature_mode( PyObject* self, PyObject* )
{
  Pds::UsdUsb::ConfigV1* obj = pypdsdata::UsdUsb::ConfigV1::pdsObject(self);
  if(not obj) return 0;

  PyObject* list = PyList_New(Pds::UsdUsb::ConfigV1::NCHANNELS);

  // copy coordinates to the list
  for ( unsigned i = 0; i < Pds::UsdUsb::ConfigV1::NCHANNELS; ++ i ) {
    PyObject* eobj = quadModeEnum.Enum_FromLong(obj->quadrature_mode(i));
    PyList_SET_ITEM( list, i, eobj );
  }

  return list;
}

PyObject*
_repr( PyObject *self )
{
  Pds::UsdUsb::ConfigV1* obj = pypdsdata::UsdUsb::ConfigV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "usdusb.ConfigV1(counting_modes=[";
  for (int i = 0; i != Pds::UsdUsb::ConfigV1::NCHANNELS; ++ i) {
    if (i != 0) str << ", ";
    str << Pds::UsdUsb::ConfigV1::count_mode_labels()[obj->counting_mode(i)];
  }
  str << "], quadrature_modes=[";
  for (int i = 0; i != Pds::UsdUsb::ConfigV1::NCHANNELS; ++ i) {
    if (i != 0) str << ", ";
    str << Pds::UsdUsb::ConfigV1::quad_mode_labels()[obj->quadrature_mode(i)];
  }
  str << "])" ;

  return PyString_FromString( str.str().c_str() );
}

}
