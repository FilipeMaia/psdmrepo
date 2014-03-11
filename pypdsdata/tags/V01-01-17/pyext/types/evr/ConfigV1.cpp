//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_ConfigV1...
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
#include "../../Exception.h"
#include "OutputMap.h"
#include "PulseConfig.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV1, npulses)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV1, pulses)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV1, noutputs)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV1, output_maps)
  PyObject* pulse( PyObject* self, PyObject* args );
  PyObject* output_map( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    { "npulses",    npulses,     METH_NOARGS,  "self.npulses() -> int\n\nReturns number of pulse configurations" },
    { "pulses",     pulses,      METH_NOARGS,  "self.pulses() -> list\n\nReturns list of :py:class:`PulseConfig` objects" },
    { "pulse",      pulse,       METH_VARARGS, "self.pulse(i: int) -> PulseConfig\n\nReturns pulse configuration (:py:class:`PulseConfig`)" },
    { "noutputs",   noutputs,    METH_NOARGS,  "self.noutputs() -> int\n\nReturns number of output configurations" },
    { "output_maps",output_maps, METH_NOARGS,  "self.output_maps() -> list\n\nReturns list of :py:class:`OutputMap` objects" },
    { "output_map", output_map,  METH_VARARGS, "self.output_map(i: int) -> OutputMap\n\nReturns output configuration (:py:class:`OutputMap`)" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::ConfigV1 class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::EvrData::ConfigV1::print(std::ostream& str) const
{
  str << "evr.ConfigV1(";

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
    str << output_maps[i].map();
  }
  str << "]";

  str << ")";
}

namespace {

PyObject*
pulse( PyObject* self, PyObject* args )
{
  const Pds::EvrData::ConfigV1* obj = pypdsdata::EvrData::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:EvrData.ConfigV1.pulse", &idx ) ) return 0;

  return pypdsdata::EvrData::PulseConfig::PyObject_FromPds( obj->pulses()[idx] );
}

PyObject*
output_map( PyObject* self, PyObject* args )
{
  const Pds::EvrData::ConfigV1* obj = pypdsdata::EvrData::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:EvrData.ConfigV1.output_map", &idx ) ) return 0;

  return pypdsdata::EvrData::OutputMap::PyObject_FromPds( obj->output_maps()[idx] );
}

}
