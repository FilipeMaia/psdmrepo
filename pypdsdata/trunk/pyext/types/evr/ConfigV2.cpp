//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_ConfigV2...
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
#include "OutputMap.h"
#include "PulseConfig.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  pypdsdata::EnumType::Enum ratecodeEnumValues[] = {
      { "r120Hz",         Pds::EvrData::ConfigV2::r120Hz },
      { "r60Hz",          Pds::EvrData::ConfigV2::r60Hz },
      { "r30Hz",          Pds::EvrData::ConfigV2::r30Hz },
      { "r10Hz",          Pds::EvrData::ConfigV2::r10Hz },
      { "r5Hz",           Pds::EvrData::ConfigV2::r5Hz },
      { "r1Hz",           Pds::EvrData::ConfigV2::r1Hz },
      { "r0_5Hz",         Pds::EvrData::ConfigV2::r0_5Hz },
      { "Single",         Pds::EvrData::ConfigV2::Single },
      { "NumberOfRates",  Pds::EvrData::ConfigV2::NumberOfRates },
      { 0, 0 }
  };
  pypdsdata::EnumType ratecodeEnum ( "RateCode", ratecodeEnumValues );

  pypdsdata::EnumType::Enum beamcodeEnumValues[] = {
      { "Off",   Pds::EvrData::ConfigV2::Off },
      { "On",    Pds::EvrData::ConfigV2::On },
      { 0, 0 }
  };
  pypdsdata::EnumType beamcodeEnum ( "BeamCode", beamcodeEnumValues );

  // type-specific methods
  ENUM_FUN0_WRAPPER(pypdsdata::EvrData::ConfigV2, beam, beamcodeEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::EvrData::ConfigV2, rate, ratecodeEnum)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV2, opcode)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV2, npulses)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV2, noutputs)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV2, size)
  PyObject* pulse( PyObject* self, PyObject* args );
  PyObject* output_map( PyObject* self, PyObject* args );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "beam",       beam,        METH_NOARGS, "" },
    { "rate",       rate,        METH_NOARGS, "" },
    { "opcode",     opcode,      METH_NOARGS, "" },
    { "npulses",    npulses,     METH_NOARGS, "pulse configurations appended to this structure" },
    { "pulse",      pulse,       METH_VARARGS, "pulse configurations appended to this structure" },
    { "noutputs",   noutputs,    METH_NOARGS, "output configurations appended to this structure" },
    { "output_map", output_map,  METH_VARARGS, "output configurations appended to this structure" },
    { "size",       size,        METH_NOARGS, "size including appended PulseConfig's and OutputMap's" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::ConfigV2 class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::ConfigV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "RateCode", ratecodeEnum.type() );
  PyDict_SetItemString( tp_dict, "BeamCode", beamcodeEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "ConfigV2", module );
}

namespace {

PyObject*
pulse( PyObject* self, PyObject* args )
{
  const Pds::EvrData::ConfigV2* obj = pypdsdata::EvrData::ConfigV2::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:EvrData.ConfigV2.pulse", &idx ) ) return 0;

  return pypdsdata::EvrData::PulseConfig::PyObject_FromPds( obj->pulse(idx) );
}

PyObject*
output_map( PyObject* self, PyObject* args )
{
  const Pds::EvrData::ConfigV2* obj = pypdsdata::EvrData::ConfigV2::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:EvrData.ConfigV2.output_map", &idx ) ) return 0;

  return pypdsdata::EvrData::OutputMap::PyObject_FromPds( obj->output_map(idx) );
}

PyObject*
_repr( PyObject *self )
{
  Pds::EvrData::ConfigV2* obj = pypdsdata::EvrData::ConfigV2::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "evr.ConfigV2(beam=" << obj->beam()
      << ", rate=" << obj->rate(); 

  str << ", pulses=["; 
  for (unsigned i = 0; i != obj->npulses(); ++ i ) {
    if (i != 0) str << ", ";
    str << obj->pulse(i).pulse();
  }
  str << "]";

  str << ", outputs=["; 
  for (unsigned i = 0; i != obj->noutputs(); ++ i ) {
    if (i != 0) str << ", ";
    str << obj->output_map(i).map();
  }
  str << "]";

  str << ")";
  return PyString_FromString( str.str().c_str() );
}

}
