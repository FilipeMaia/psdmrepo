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
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV2, pulses)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV2, noutputs)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV2, output_maps)
  PyObject* pulse( PyObject* self, PyObject* args );
  PyObject* output_map( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    { "beam",       beam,        METH_NOARGS, "self.beam() -> BeamCode enum\n\nReturns :py:class:`BeamCode` enum" },
    { "rate",       rate,        METH_NOARGS, "self.rate() -> RateCodeenum\n\nReturns :py:class:`RateCode` enum" },
    { "opcode",     opcode,      METH_NOARGS, "self.opcode() -> int\n\nReturns integer number" },
    { "npulses",    npulses,     METH_NOARGS,  "self.npulses() -> int\n\nReturns number of pulse configurations" },
    { "pulses",     pulses,      METH_NOARGS,  "self.pulses() -> list\n\nReturns list of :py:class:`PulseConfig` objects" },
    { "pulse",      pulse,       METH_VARARGS, "self.pulse(i: int) -> PulseConfig\n\nReturns pulse configuration (:py:class:`PulseConfig`)" },
    { "noutputs",   noutputs,    METH_NOARGS,  "self.noutputs() -> int\n\nReturns number of output configurations" },
    { "output_maps",output_maps, METH_NOARGS,  "self.output_maps() -> list\n\nReturns list of :py:class:`OutputMap` objects" },
    { "output_map", output_map,  METH_VARARGS, "self.output_map(i: int) -> OutputMap\n\nReturns output configuration (:py:class:`OutputMap`)" },
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

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "RateCode", ratecodeEnum.type() );
  PyDict_SetItemString( tp_dict, "BeamCode", beamcodeEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "ConfigV2", module );
}

void
pypdsdata::EvrData::ConfigV2::print(std::ostream& str) const
{
  str << "evr.ConfigV2(beam=" << m_obj->beam()
      << ", rate=" << m_obj->rate();

  str << "pulses=[";
  const ndarray<const Pds::EvrData::PulseConfig, 1>& pulses = m_obj->pulses();
  for (unsigned i = 0; i != pulses.size(); ++ i ) {
    if (i != 0) str << ", ";
    str << pulses[i].pulse();
  }
  str << "]";

  str << ", outputs=[";
  const ndarray<const Pds::EvrData::OutputMap, 1>& output_maps = m_obj->output_maps();
  for (unsigned i = 0; i != output_maps.size(); ++ i ) {
    if (i != 0) str << ", ";
    str << output_maps[i].value();
  }
  str << "]";

  str << ")";
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

  return pypdsdata::EvrData::PulseConfig::PyObject_FromPds( obj->pulses()[idx] );
}

PyObject*
output_map( PyObject* self, PyObject* args )
{
  const Pds::EvrData::ConfigV2* obj = pypdsdata::EvrData::ConfigV2::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:EvrData.ConfigV2.output_map", &idx ) ) return 0;

  return pypdsdata::EvrData::OutputMap::PyObject_FromPds( obj->output_maps()[idx] );
}

}
