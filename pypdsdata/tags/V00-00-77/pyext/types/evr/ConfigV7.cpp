//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_ConfigV7...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV7.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "OutputMapV2.h"
#include "EventCodeV6.h"
#include "PulseConfigV3.h"
#include "SequencerConfigV1.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV7, neventcodes)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV7, npulses)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV7, noutputs)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV7, size)
  PyObject* eventcode( PyObject* self, PyObject* args );
  PyObject* pulse( PyObject* self, PyObject* args );
  PyObject* output_map( PyObject* self, PyObject* args );
  PyObject* seq_config( PyObject* self, PyObject*);
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "neventcodes",neventcodes, METH_NOARGS,  "self.neventcodes() -> int\n\nReturns number of event codes" },
    { "eventcode",  eventcode,   METH_VARARGS, "self.eventcode(i: int) -> EventCodeV6\n\nReturns event code (:py:class:`EventCodeV6` object)" },
    { "npulses",    npulses,     METH_NOARGS,  "self.npulses() -> int\n\nReturns number of pulse configurations" },
    { "pulse",      pulse,       METH_VARARGS, "self.pulse(i: int) -> PulseConfigV3\n\nReturns pulse configuration (:py:class:`PulseConfigV3`)" },
    { "noutputs",   noutputs,    METH_NOARGS,  "self.noutputs() -> int\n\nReturns number of output configurations" },
    { "output_map", output_map,  METH_VARARGS, "self.output_map(i: int) -> OutputMapV2\n\nReturns output configuration (:py:class:`OutputMapV2`)" },
    { "seq_config", seq_config,  METH_NOARGS,  "self.seq_config() -> SequencerConfigV1\n\nReturns :py:class:`SequencerConfigV1` object" },
    { "size",       size,        METH_NOARGS,  "self.size() -> int\n\nRetuns structure size" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::ConfigV7 class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::ConfigV7::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "ConfigV7", module );
}

namespace {

PyObject*
eventcode( PyObject* self, PyObject* args )
{
  const Pds::EvrData::ConfigV7* obj = pypdsdata::EvrData::ConfigV7::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:EvrData.ConfigV7.eventcode", &idx ) ) return 0;

  return pypdsdata::EvrData::EventCodeV6::PyObject_FromPds( obj->eventcode(idx) );
}

PyObject*
pulse( PyObject* self, PyObject* args )
{
  const Pds::EvrData::ConfigV7* obj = pypdsdata::EvrData::ConfigV7::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:EvrData.ConfigV7.pulse", &idx ) ) return 0;

  return pypdsdata::EvrData::PulseConfigV3::PyObject_FromPds( obj->pulse(idx) );
}

PyObject*
output_map( PyObject* self, PyObject* args )
{
  const Pds::EvrData::ConfigV7* obj = pypdsdata::EvrData::ConfigV7::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:EvrData.ConfigV7.output_map", &idx ) ) return 0;

  return pypdsdata::EvrData::OutputMapV2::PyObject_FromPds( obj->output_map(idx) );
}

PyObject*
seq_config( PyObject* self, PyObject* )
{
  const Pds::EvrData::ConfigV7* obj = pypdsdata::EvrData::ConfigV7::pdsObject( self );
  if ( not obj ) return 0;

  Pds::EvrData::SequencerConfigV1& seq_config = const_cast<Pds::EvrData::SequencerConfigV1&>(obj->seq_config());
  return pypdsdata::EvrData::SequencerConfigV1::PyObject_FromPds( &seq_config, self, seq_config.size() );
}

PyObject*
_repr( PyObject *self )
{
  Pds::EvrData::ConfigV7* pdsObj = pypdsdata::EvrData::ConfigV7::pdsObject(self);
  if(not pdsObj) return 0;

  std::ostringstream str;
  str << "evr.ConfigV7(";

  str << "eventcodes=["; 
  for (unsigned i = 0; i != pdsObj->neventcodes(); ++ i ) {
    if (i != 0) str << ", ";
    str << pdsObj->eventcode(i).code();
  }
  str << "]";

  str << ", pulses=["; 
  for (unsigned i = 0; i != pdsObj->npulses(); ++ i ) {
    if (i != 0) str << ", ";
    str << pdsObj->pulse(i).pulseId();
  }
  str << "]";

  str << ", outputs=["; 
  for (unsigned i = 0; i != pdsObj->noutputs(); ++ i ) {
    if (i != 0) str << ", ";
    str << pdsObj->output_map(i).map();
  }
  str << "]";

  str << ")";
  return PyString_FromString( str.str().c_str() );
}

}
