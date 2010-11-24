//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

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
#include "ConfigV1QuadReg.h"
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
  PyObject* sections( PyObject* self, PyObject* args);
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV2, tdi)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV2, quadMask)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV2, runDelay)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV2, eventCode)
  ENUM_FUN0_WRAPPER(pypdsdata::CsPad::ConfigV2, inactiveRunMode, runModesEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::CsPad::ConfigV2, activeRunMode, runModesEnum)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV2, payloadSize)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV2, badAsicMask0)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV2, badAsicMask1)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV2, asicMask)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV2, numAsicsRead)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV2, concentratorVersion)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"quads",               quads,               METH_NOARGS, "" },
    {"numQuads",            numQuads,            METH_NOARGS, "" },
    {"tdi",                 tdi,                 METH_NOARGS, "" },
    {"quadMask",            quadMask,            METH_NOARGS, "" },
    {"runDelay",            runDelay,            METH_NOARGS, "" },
    {"eventCode",           eventCode,           METH_NOARGS, "" },
    {"inactiveRunMode",     inactiveRunMode,     METH_NOARGS, "" },
    {"activeRunMode",       activeRunMode,       METH_NOARGS, "" },
    {"payloadSize",         payloadSize,         METH_NOARGS, "" },
    {"badAsicMask0",        badAsicMask0,        METH_NOARGS, "" },
    {"badAsicMask1",        badAsicMask1,        METH_NOARGS, "" },
    {"asicMask",            asicMask,            METH_NOARGS, "" },
    {"roiMask",             roiMask,             METH_VARARGS, "" },
    {"numAsicsRead",        numAsicsRead,        METH_NOARGS, "" },
    {"numAsicsStored",      numAsicsStored,      METH_VARARGS, "" },
    {"concentratorVersion", concentratorVersion, METH_NOARGS, "" },
    {"sections",            sections,            METH_VARARGS, "list of sections read for a given quadrant number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad::ConfigV2 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad::ConfigV2::initType( PyObject* module )
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

  BaseType::initType( "ConfigV2", module );
}

namespace {

PyObject*
quads( PyObject* self, PyObject* )
{
  Pds::CsPad::ConfigV2* obj = pypdsdata::CsPad::ConfigV2::pdsObject( self );
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
  const Pds::CsPad::ConfigV2* obj = pypdsdata::CsPad::ConfigV2::pdsObject( self );
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
  const Pds::CsPad::ConfigV2* obj = pypdsdata::CsPad::ConfigV2::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned q ;
  if ( not PyArg_ParseTuple( args, "I:cspad.ConfigV2.roiMask", &q ) ) return 0;
  
  if ( q >= Pds::CsPad::MaxQuadsPerSensor ) {
    PyErr_SetString(PyExc_IndexError, "index outside of range [0..MaxQuadsPerSensor) in cspad.ConfigV2.roiMask()");
    return 0;
  }

  return PyInt_FromLong( obj->roiMask(q) );
}

PyObject*
numAsicsStored( PyObject* self, PyObject* args )
{
  const Pds::CsPad::ConfigV2* obj = pypdsdata::CsPad::ConfigV2::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned q ;
  if ( not PyArg_ParseTuple( args, "I:cspad.ConfigV2.numAsicsStored", &q ) ) return 0;
  
  if ( q >= Pds::CsPad::MaxQuadsPerSensor ) {
    PyErr_SetString(PyExc_IndexError, "index outside of range [0..MaxQuadsPerSensor) in cspad.ConfigV2.numAsicsStored()");
    return 0;
  }

  return PyInt_FromLong( obj->numAsicsStored(q) );
}

PyObject*
sections( PyObject* self, PyObject* args )
{
  const Pds::CsPad::ConfigV2* obj = pypdsdata::CsPad::ConfigV2::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned q ;
  if ( not PyArg_ParseTuple( args, "I:cspad.ConfigV2.sections", &q ) ) return 0;
  
  if ( q >= Pds::CsPad::MaxQuadsPerSensor ) {
    PyErr_SetString(PyExc_IndexError, "index outside of range [0..MaxQuadsPerSensor) in cspad.ConfigV2.sections()");
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
  Pds::CsPad::ConfigV2* obj = pypdsdata::CsPad::ConfigV2::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "cspad.ConfigV2(quadMask=" << obj->quadMask()
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
