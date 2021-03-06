//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_ConfigV4...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV4.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "OutputMap.h"
#include "EventCodeV4.h"
#include "PulseConfigV3.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV4, neventcodes)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV4, npulses)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV4, noutputs)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV4, size)
  PyObject* eventcode( PyObject* self, PyObject* args );
  PyObject* pulse( PyObject* self, PyObject* args );
  PyObject* output_map( PyObject* self, PyObject* args );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "neventcodes",neventcodes, METH_NOARGS,  "self.neventcodes() -> int\n\nReturns number of event codes" },
    { "eventcode",  eventcode,   METH_VARARGS, "self.eventcode(i: int) -> EventCodeV4\n\nReturns event code (:py:class:`EventCodeV4` object)" },
    { "npulses",    npulses,     METH_NOARGS,  "self.npulses() -> int\n\nReturns number of pulse configurations" },
    { "pulse",      pulse,       METH_VARARGS, "self.pulse(i: int) -> PulseConfigV3\n\nReturns pulse configuration (:py:class:`PulseConfigV3`)" },
    { "noutputs",   noutputs,    METH_NOARGS,  "self.noutputs() -> int\n\nReturns number of output configurations" },
    { "output_map", output_map,  METH_VARARGS, "self.output_map(i: int) -> OutputMap\n\nReturns output configuration (:py:class:`OutputMap`)" },
    { "size",       size,        METH_NOARGS,  "self.size() -> int\n\nRetuns structure size" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::ConfigV4 class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::ConfigV4::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "ConfigV4", module );
}

namespace {

PyObject*
eventcode( PyObject* self, PyObject* args )
{
  const Pds::EvrData::ConfigV4* obj = pypdsdata::EvrData::ConfigV4::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:EvrData.ConfigV4.eventcode", &idx ) ) return 0;

  return pypdsdata::EvrData::EventCodeV4::PyObject_FromPds( obj->eventcode(idx) );
}

PyObject*
pulse( PyObject* self, PyObject* args )
{
  const Pds::EvrData::ConfigV4* obj = pypdsdata::EvrData::ConfigV4::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:EvrData.ConfigV4.pulse", &idx ) ) return 0;

  return pypdsdata::EvrData::PulseConfigV3::PyObject_FromPds( obj->pulse(idx) );
}

PyObject*
output_map( PyObject* self, PyObject* args )
{
  const Pds::EvrData::ConfigV4* obj = pypdsdata::EvrData::ConfigV4::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:EvrData.ConfigV4.output_map", &idx ) ) return 0;

  return pypdsdata::EvrData::OutputMap::PyObject_FromPds( obj->output_map(idx) );
}

PyObject*
_repr( PyObject *self )
{
  Pds::EvrData::ConfigV4* pdsObj = pypdsdata::EvrData::ConfigV4::pdsObject(self);
  if(not pdsObj) return 0;

  std::ostringstream str;
  str << "evr.ConfigV4(";

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
