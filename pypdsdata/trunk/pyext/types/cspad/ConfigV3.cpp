//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV3...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV3.h"

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
      { "NoRunning",                Pds::CsPad::NoRunning },
      { "RunButDrop",               Pds::CsPad::RunButDrop },
      { "RunAndSendToRCE",          Pds::CsPad::RunAndSendToRCE },
      { "RunAndSendTriggeredByTTL", Pds::CsPad::RunAndSendTriggeredByTTL },
      { "ExternalTriggerSendToRCE", Pds::CsPad::ExternalTriggerSendToRCE },
      { "ExternalTriggerDrop",      Pds::CsPad::ExternalTriggerDrop },
      { "NumberOfRunModes",         Pds::CsPad::NumberOfRunModes },
      { 0, 0 }
  };
  pypdsdata::EnumType runModesEnum ( "RunModes", runModesEnumValues );

  // methods
  PyObject* quads( PyObject* self, PyObject* );
  PyObject* numQuads( PyObject* self, PyObject* );
  PyObject* roiMask( PyObject* self, PyObject* args);
  PyObject* numAsicsStored( PyObject* self, PyObject* args);
  PyObject* protectionThresholds( PyObject* self, PyObject*);
  PyObject* sections( PyObject* self, PyObject* args);
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV3, tdi)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV3, quadMask)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV3, runDelay)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV3, eventCode)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV3, protectionEnable)
  ENUM_FUN0_WRAPPER(pypdsdata::CsPad::ConfigV3, inactiveRunMode, runModesEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::CsPad::ConfigV3, activeRunMode, runModesEnum)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV3, payloadSize)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV3, badAsicMask0)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV3, badAsicMask1)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV3, asicMask)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV3, numAsicsRead)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV3, concentratorVersion)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"quads",               quads,               METH_NOARGS, "self.quads() -> list\n\nReturns list of ConfigV1QuadReg objects" },
    {"numQuads",            numQuads,            METH_NOARGS, "self.numQuads() -> int\n\nReturns number of quadrants" },
    {"tdi",                 tdi,                 METH_NOARGS, "self.tdi() -> int\n\nReturns test data index" },
    {"quadMask",            quadMask,            METH_NOARGS, "self.quadMask() -> int\n\nReturns quadrant bit mask" },
    {"runDelay",            runDelay,            METH_NOARGS, "self.runDelay() -> int\n\nReturns integer number" },
    {"eventCode",           eventCode,           METH_NOARGS, "self.eventCode() -> int\n\nReturns event code" },
    {"protectionThresholds", protectionThresholds, METH_NOARGS,
        "self.protectionThresholds() -> list\n\nReturns list of MaxQuadsPerSensor objects of type CsPadProtectionSystemThreshold" },
    {"protectionEnable",    protectionEnable,    METH_NOARGS, "self.protectionEnable() -> int\n\nReturns integer number" },
    {"inactiveRunMode",     inactiveRunMode,     METH_NOARGS, "self.inactiveRunMode() -> RunModes enum\n\nReturns RunModes enum" },
    {"activeRunMode",       activeRunMode,       METH_NOARGS, "self.activeRunMode() -> RunModes enum\n\nReturns RunModes enum" },
    {"payloadSize",         payloadSize,         METH_NOARGS, "self.payloadSize() -> int\n\nReturns size of data" },
    {"badAsicMask0",        badAsicMask0,        METH_NOARGS, "self.badAsicMask0() -> int\n\nRetuns bit mask" },
    {"badAsicMask1",        badAsicMask1,        METH_NOARGS, "self.badAsicMask1() -> int\n\nRetuns bit mask" },
    {"asicMask",            asicMask,            METH_NOARGS, "self.asicMask() -> int\n\nRetuns bit mask" },
    {"roiMask",             roiMask,             METH_VARARGS, "self.roiMask(q: int) -> int\n\nRetuns sections bit mask for given quadrant" },
    {"numAsicsRead",        numAsicsRead,        METH_NOARGS, "self.numAsicsRead() -> int\n\nRetuns number of ASICs in readout" },
    {"numAsicsStored",      numAsicsStored,      METH_VARARGS,
        "self.numAsicsStored(q: int) -> int\n\nRetuns number of ASICs stored for a given quadrant" },
    {"concentratorVersion", concentratorVersion, METH_NOARGS, "self.concentratorVersion() -> int\n\nReturns concentrator version" },
    {"sections",            sections,            METH_VARARGS,
        "self.sections(q: int) -> list of int\n\nlist of section indices read for a given quadrant number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad::ConfigV3 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad::ConfigV3::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "RunModes", runModesEnum.type() );
  PyObject* val = PyInt_FromLong(Pds::CsPad::MaxQuadsPerSensor);
  PyDict_SetItemString( type->tp_dict, "MaxQuadsPerSensor", val );
  Py_XDECREF(val);

  BaseType::initType( "ConfigV3", module );
}

namespace {

PyObject*
quads( PyObject* self, PyObject* )
{
  Pds::CsPad::ConfigV3* obj = pypdsdata::CsPad::ConfigV3::pdsObject( self );
  if ( not obj ) return 0;

  PyObject* list = PyList_New( Pds::CsPad::MaxQuadsPerSensor );
  Pds::CsPad::ConfigV1QuadReg* quads = obj->quads();
  for ( unsigned i = 0 ; i < Pds::CsPad::MaxQuadsPerSensor ; ++ i ) {
    PyObject* q = pypdsdata::CsPad::ConfigV1QuadReg::PyObject_FromPds(
        &quads[i], self, sizeof(Pds::CsPad::ConfigV1QuadReg) );
    PyList_SET_ITEM( list, i, q );
  }

  return list;
}

PyObject*
numQuads( PyObject* self, PyObject* )
{
  const Pds::CsPad::ConfigV3* obj = pypdsdata::CsPad::ConfigV3::pdsObject( self );
  if ( not obj ) return 0;

  unsigned count = 0 ;
  unsigned mask = obj->quadMask();
  for ( unsigned i = Pds::CsPad::MaxQuadsPerSensor ; i ; -- i ) {
    if ( mask & 1 ) ++ count ;
    mask >>= 1 ;
  }
  
  return PyInt_FromLong(count);
}

PyObject*
roiMask( PyObject* self, PyObject* args )
{
  const Pds::CsPad::ConfigV3* obj = pypdsdata::CsPad::ConfigV3::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned q ;
  if ( not PyArg_ParseTuple( args, "I:cspad.ConfigV3.roiMask", &q ) ) return 0;
  
  if ( q >= Pds::CsPad::MaxQuadsPerSensor ) {
    PyErr_SetString(PyExc_IndexError, "index outside of range [0..MaxQuadsPerSensor) in cspad.ConfigV3.roiMask()");
    return 0;
  }

  return PyInt_FromLong( obj->roiMask(q) );
}

PyObject*
numAsicsStored( PyObject* self, PyObject* args )
{
  const Pds::CsPad::ConfigV3* obj = pypdsdata::CsPad::ConfigV3::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned q ;
  if ( not PyArg_ParseTuple( args, "I:cspad.ConfigV3.numAsicsStored", &q ) ) return 0;
  
  if ( q >= Pds::CsPad::MaxQuadsPerSensor ) {
    PyErr_SetString(PyExc_IndexError, "index outside of range [0..MaxQuadsPerSensor) in cspad.ConfigV3.numAsicsStored()");
    return 0;
  }

  return PyInt_FromLong( obj->numAsicsStored(q) );
}

PyObject*
protectionThresholds( PyObject* self, PyObject* )
{
  Pds::CsPad::ConfigV3* obj = pypdsdata::CsPad::ConfigV3::pdsObject( self );
  if ( not obj ) return 0;

  PyObject* list = PyList_New( Pds::CsPad::MaxQuadsPerSensor );
  Pds::CsPad::ProtectionSystemThreshold* thresholds = obj->protectionThresholds();
  for ( unsigned i = 0 ; i < Pds::CsPad::MaxQuadsPerSensor ; ++ i ) {
    PyObject* q = pypdsdata::CsPad::CsPadProtectionSystemThreshold::PyObject_FromPds(
        &thresholds[i], self, sizeof(Pds::CsPad::ProtectionSystemThreshold) );
    PyList_SET_ITEM( list, i, q );
  }

  return list;
}

PyObject*
sections( PyObject* self, PyObject* args )
{
  const Pds::CsPad::ConfigV3* obj = pypdsdata::CsPad::ConfigV3::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned q ;
  if ( not PyArg_ParseTuple( args, "I:cspad.ConfigV3.sections", &q ) ) return 0;
  
  if ( q >= Pds::CsPad::MaxQuadsPerSensor ) {
    PyErr_SetString(PyExc_IndexError, "index outside of range [0..MaxQuadsPerSensor) in cspad.ConfigV3.sections()");
    return 0;
  }

  unsigned mask = obj->roiMask(q);
  unsigned count = 0;
  for ( unsigned i = 0 ; i < sizeof(mask)*8 ; ++ i ) {
    if (mask & (1<<i)) ++ count;
  }
  PyObject* list = PyList_New( count );
  unsigned ic = 0 ;
  for ( unsigned i = 0 ; i < sizeof(mask)*8 ; ++ i ) {
    if (mask & (1<<i)) {
      PyList_SET_ITEM( list, ic++, pypdsdata::TypeLib::toPython(i) );
    }
  }
  return list;
}

PyObject*
_repr( PyObject *self )
{
  Pds::CsPad::ConfigV3* obj = pypdsdata::CsPad::ConfigV3::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "cspad.ConfigV3(quadMask=" << obj->quadMask()
      << ", eventCode=" << obj->eventCode()
      << ", asicMask=" << obj->asicMask()
      << ", numAsicsRead=" << obj->numAsicsRead();
  
  str << ", numAsicsStored=[";
  for (int q = 0; q < 4; ++ q ) {
    if (q) str << ", ";
    str << obj->numAsicsStored(q);
  }
  str << "]";

  str << ", roiMask=[";
  for (int q = 0; q < 4; ++ q ) {
    if (q) str << ", ";
    str << obj->roiMask(q);
  }
  str << "]";
  
  str << ")";

  return PyString_FromString( str.str().c_str() );
}

}
