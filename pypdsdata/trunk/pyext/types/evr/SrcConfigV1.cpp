//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_SrcConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "SrcConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "OutputMapV2.h"
#include "SrcEventCode.h"
#include "PulseConfigV3.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::EvrData::SrcConfigV1, neventcodes)
  FUN0_WRAPPER(pypdsdata::EvrData::SrcConfigV1, eventcodes)
  FUN0_WRAPPER(pypdsdata::EvrData::SrcConfigV1, npulses)
  FUN0_WRAPPER(pypdsdata::EvrData::SrcConfigV1, pulses)
  FUN0_WRAPPER(pypdsdata::EvrData::SrcConfigV1, noutputs)
  FUN0_WRAPPER(pypdsdata::EvrData::SrcConfigV1, output_maps)

  PyMethodDef methods[] = {
    { "neventcodes",neventcodes, METH_NOARGS,  "self.neventcodes() -> int\n\nReturns number of event codes" },
    { "eventcodes", eventcodes,  METH_NOARGS,  "self.eventcodes() -> list\n\nReturns list of :py:class:`SrcEventCode` objects" },
    { "npulses",    npulses,     METH_NOARGS,  "self.npulses() -> int\n\nReturns number of pulse configurations" },
    { "pulses",     pulses,      METH_NOARGS,  "self.pulses() -> list\n\nReturns list of :py:class:`PulseConfigV3` objects" },
    { "noutputs",   noutputs,    METH_NOARGS,  "self.noutputs() -> int\n\nReturns number of output configurations" },
    { "output_maps",output_maps, METH_NOARGS,  "self.output_maps() -> list\n\nReturns list of :py:class:`OutputMapV2` objects" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::SrcConfigV1 class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::SrcConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "SrcConfigV1", module );
}

void
pypdsdata::EvrData::SrcConfigV1::print(std::ostream& str) const
{
  str << "evr.SrcConfigV1(";

  str << "eventcodes=[";
  const ndarray<const Pds::EvrData::SrcEventCode, 1>& eventcodes = m_obj->eventcodes();
  for (unsigned i = 0; i != eventcodes.size(); ++ i ) {
    if (i != 0) str << ", ";
    str << eventcodes[i].code();
  }
  str << "]";

  str << ", pulses=[";
  const ndarray<const Pds::EvrData::PulseConfigV3, 1>& pulses = m_obj->pulses();
  for (unsigned i = 0; i != pulses.size(); ++ i ) {
    if (i != 0) str << ", ";
    str << pulses[i].pulseId();
  }
  str << "]";

  str << ", outputs=[";
  const ndarray<const Pds::EvrData::OutputMapV2, 1>& output_maps = m_obj->output_maps();
  for (unsigned i = 0; i != output_maps.size(); ++ i ) {
    if (i != 0) str << ", ";
    str << output_maps[i].map();
  }
  str << "]";

  str << ")";
}
