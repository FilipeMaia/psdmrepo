//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Timepix_TM6740ConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV2.h"

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
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  pypdsdata::EnumType::Enum speedEnumValues[] = {
      { "ReadoutSpeed_Slow",  Pds::Timepix::ConfigV2::ReadoutSpeed_Slow },
      { "ReadoutSpeed_Fast",  Pds::Timepix::ConfigV2::ReadoutSpeed_Fast },
      { 0, 0 }
  };
  pypdsdata::EnumType speedEnum ( "ReadoutSpeed", speedEnumValues );

  pypdsdata::EnumType::Enum trigEnumValues[] = {
      { "TriggerMode_ExtPos", Pds::Timepix::ConfigV2::TriggerMode_ExtPos },
      { "TriggerMode_ExtNeg", Pds::Timepix::ConfigV2::TriggerMode_ExtNeg },
      { "TriggerMode_Soft",   Pds::Timepix::ConfigV2::TriggerMode_Soft },
      { 0, 0 }
  };
  pypdsdata::EnumType trigEnum ( "TriggerMode", trigEnumValues );


  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
        { "ChipCount", Pds::Timepix::ConfigV2::ChipCount },
        { "ChipNameMax", Pds::Timepix::ConfigV2::ChipNameMax },
        { "PixelThreshMax", Pds::Timepix::ConfigV2::PixelThreshMax },
        { 0, 0 }
  };

  // methods
  ENUM_FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, readoutSpeed, speedEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, triggerMode, trigEnum)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, timepixSpeed)

  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac0Ikrum)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac0Disc)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac0Preamp)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac0BufAnalogA)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac0BufAnalogB)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac0Hist)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac0ThlFine)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac0ThlCourse)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac0Vcas)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac0Fbk)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac0Gnd)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac0Ths)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac0BiasLvds)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac0RefLvds)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac1Ikrum)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac1Disc)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac1Preamp)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac1BufAnalogA)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac1BufAnalogB)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac1Hist)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac1ThlFine)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac1ThlCourse)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac1Vcas)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac1Fbk)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac1Gnd)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac1Ths)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac1BiasLvds)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac1RefLvds)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac2Ikrum)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac2Disc)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac2Preamp)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac2BufAnalogA)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac2BufAnalogB)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac2Hist)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac2ThlFine)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac2ThlCourse)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac2Vcas)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac2Fbk)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac2Gnd)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac2Ths)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac2BiasLvds)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac2RefLvds)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac3Ikrum)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac3Disc)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac3Preamp)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac3BufAnalogA)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac3BufAnalogB)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac3Hist)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac3ThlFine)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac3ThlCourse)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac3Vcas)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac3Fbk)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac3Gnd)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac3Ths)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac3BiasLvds)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, dac3RefLvds)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, chipCount)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, driverVersion)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, firmwareVersion)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, pixelThreshSize)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, chip0Name)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, chip1Name)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, chip2Name)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, chip3Name)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, chip0ID)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, chip1ID)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, chip2ID)
  FUN0_WRAPPER(pypdsdata::Timepix::ConfigV2, chip3ID)
  PyObject* pixelThresh(PyObject *self, PyObject*);
  PyObject* _repr(PyObject *self);

  PyMethodDef methods[] = {
    { "readoutSpeed",      readoutSpeed        , METH_NOARGS,  "self.readoutSpeed() -> ReadoutSpeed enum\n\nReturns ReadoutSpeed enum" },
    { "triggerMode",       triggerMode         , METH_NOARGS,  "self.triggerMode() -> TriggerMode enum\n\nReturns TriggerMode enum" },
    { "timepixSpeed",      timepixSpeed        , METH_NOARGS,  "self.timepixSpeed() -> int\n\nReturns integer number" },
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
    { "chipCount",         chipCount           , METH_NOARGS,  "self.chipCount() -> int\n\nReturns number of chips" },
    { "driverVersion",     driverVersion       , METH_NOARGS,  "self.driverVersion() -> int\n\nReturns integer number" },
    { "firmwareVersion",   firmwareVersion     , METH_NOARGS,  "self.firmwareVersion() -> int\n\nReturns integer number" },
    { "pixelThreshSize",   pixelThreshSize     , METH_NOARGS,  "self.pixelThreshSize() -> int\n\nReturns integer number" },
    { "pixelThresh",       pixelThresh         , METH_NOARGS,  "self.pixelThresh() -> numpy.ndarray\n\nReturns 1-dim array of integers" },
    { "chip0Name",         chip0Name           , METH_NOARGS,  "self.chip0Name() -> string\n\nReturns chip name as string" },
    { "chip1Name",         chip1Name           , METH_NOARGS,  "self.chip1Name() -> string\n\nReturns chip name as string" },
    { "chip2Name",         chip2Name           , METH_NOARGS,  "self.chip2Name() -> string\n\nReturns chip name as string" },
    { "chip3Name",         chip3Name           , METH_NOARGS,  "self.chip3Name() -> string\n\nReturns chip name as string" },
    { "chip0ID",           chip0ID             , METH_NOARGS,  "self.chip0ID() -> int\n\nReturns integer number" },
    { "chip1ID",           chip1ID             , METH_NOARGS,  "self.chip1ID() -> int\n\nReturns integer number" },
    { "chip2ID",           chip2ID             , METH_NOARGS,  "self.chip2ID() -> int\n\nReturns integer number" },
    { "chip3ID",           chip3ID             , METH_NOARGS,  "self.chip3ID() -> int\n\nReturns integer number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Timepix::ConfigV2 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Timepix::ConfigV2::initType( PyObject* module )
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

  BaseType::initType( "ConfigV2", module );
}

namespace {

PyObject*
pixelThresh(PyObject *self, PyObject*)
{
  Pds::Timepix::ConfigV2* obj = pypdsdata::Timepix::ConfigV2::pdsObject(self);
  if(not obj) return 0;
  
  // dimensions
  npy_intp dims[1] = { Pds::Timepix::ConfigV2::PixelThreshMax };

  // NumPy type number and flags
  int typenum = NPY_UBYTE;
  int flags = NPY_C_CONTIGUOUS ;

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 1, dims, typenum, 0,
                                (void*)obj->pixelThresh(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

PyObject*
_repr( PyObject *self )
{
  Pds::Timepix::ConfigV2* obj = pypdsdata::Timepix::ConfigV2::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "Timepix.ConfigV2(readoutSpeed=" << int(obj->readoutSpeed())
      << ", triggerMode=" << int(obj->triggerMode())
      << ", timepixSpeed=" << obj->timepixSpeed()
      << ", ...)" ;

  return PyString_FromString( str.str().c_str() );
}

}
