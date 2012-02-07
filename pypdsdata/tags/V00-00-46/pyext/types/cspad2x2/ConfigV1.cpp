//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV1...
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
#include "ConfigV1QuadReg.h"
#include "CsPadProtectionSystemThreshold.h"
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  pypdsdata::EnumType::Enum runModesEnumValues[] = {
      { "NoRunning",                Pds::CsPad2x2::NoRunning },
      { "RunButDrop",               Pds::CsPad2x2::RunButDrop },
      { "RunAndSendToRCE",          Pds::CsPad2x2::RunAndSendToRCE },
      { "RunAndSendTriggeredByTTL", Pds::CsPad2x2::RunAndSendTriggeredByTTL },
      { "ExternalTriggerSendToRCE", Pds::CsPad2x2::ExternalTriggerSendToRCE },
      { "ExternalTriggerDrop",      Pds::CsPad2x2::ExternalTriggerDrop },
      { "NumberOfRunModes",         Pds::CsPad2x2::NumberOfRunModes },
      { 0, 0 }
  };
  pypdsdata::EnumType runModesEnum ( "RunModes", runModesEnumValues );

  // methods
  PyObject* quad( PyObject* self, PyObject* );
  PyObject* protectionThreshold( PyObject* self, PyObject*);
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV1, tdi)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV1, protectionEnable)
  ENUM_FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV1, inactiveRunMode, runModesEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV1, activeRunMode, runModesEnum)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV1, payloadSize)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV1, badAsicMask)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV1, asicMask)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV1, roiMask)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV1, numAsicsRead)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV1, numAsicsStored)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV1, concentratorVersion)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"quad",                quad,                METH_NOARGS, "" },
    {"tdi",                 tdi,                 METH_NOARGS, "" },
    {"protectionThreshold", protectionThreshold, METH_NOARGS, "" },
    {"protectionEnable",    protectionEnable,    METH_NOARGS, "" },
    {"inactiveRunMode",     inactiveRunMode,     METH_NOARGS, "" },
    {"activeRunMode",       activeRunMode,       METH_NOARGS, "" },
    {"payloadSize",         payloadSize,         METH_NOARGS, "" },
    {"badAsicMask",         badAsicMask,         METH_NOARGS, "" },
    {"asicMask",            asicMask,            METH_NOARGS, "" },
    {"roiMask",             roiMask,             METH_VARARGS, "" },
    {"numAsicsRead",        numAsicsRead,        METH_NOARGS, "" },
    {"numAsicsStored",      numAsicsStored,      METH_VARARGS, "" },
    {"concentratorVersion", concentratorVersion, METH_NOARGS, "" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad2x2::ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad2x2::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "RunModes", runModesEnum.type() );
  PyObject* val = PyInt_FromLong(Pds::CsPad2x2::QuadsPerSensor);
  PyDict_SetItemString( type->tp_dict, "QuadsPerSensor", val );
  Py_XDECREF(val);

  BaseType::initType( "ConfigV1", module );
}

namespace {

PyObject*
quad( PyObject* self, PyObject* )
{
  Pds::CsPad2x2::ConfigV1* obj = pypdsdata::CsPad2x2::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::CsPad2x2::ConfigV1QuadReg::PyObject_FromPds(
      obj->quad(), self, sizeof(Pds::CsPad2x2::ConfigV1QuadReg) );
}

PyObject*
protectionThreshold( PyObject* self, PyObject* )
{
  Pds::CsPad2x2::ConfigV1* obj = pypdsdata::CsPad2x2::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::CsPad2x2::CsPadProtectionSystemThreshold::PyObject_FromPds(
      obj->protectionThreshold(), self, sizeof(Pds::CsPad2x2::ProtectionSystemThreshold) );
}

PyObject*
_repr( PyObject *self )
{
  Pds::CsPad2x2::ConfigV1* obj = pypdsdata::CsPad2x2::ConfigV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "CsPad2x2.ConfigV1(tdi=" << obj->tdi()
      << ", inactiveRunMode=" << obj->inactiveRunMode()
      << ", activeRunMode=" << obj->activeRunMode()
      << ", payloadSize=" << obj->payloadSize()
      << ", asicMask=" << obj->asicMask()
      << ", numAsicsStored=" << obj->numAsicsStored(0)
      << ", roiMask=" << obj->roiMask(0)
      << ")";

  return PyString_FromString( str.str().c_str() );
}

}
