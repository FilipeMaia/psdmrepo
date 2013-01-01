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
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV1, noutputs)
  FUN0_WRAPPER(pypdsdata::EvrData::ConfigV1, size)
  PyObject* pulse( PyObject* self, PyObject* args );
  PyObject* output_map( PyObject* self, PyObject* args );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "npulses",    npulses,     METH_NOARGS,  "self.npulses() -> int\n\nReturns number of pulse configurations" },
    { "pulse",      pulse,       METH_VARARGS, "self.pulse(i: int) -> PulseConfig\n\nReturns pulse configuration (:py:class:`PulseConfig`)" },
    { "noutputs",   noutputs,    METH_NOARGS,  "self.noutputs() -> int\n\nReturns number of output configurations" },
    { "output_map", output_map,  METH_VARARGS, "self.output_map(i: int) -> OutputMap\n\nReturns output configuration (:py:class:`OutputMap`)" },
    { "size",       size,        METH_NOARGS,  "self.size() -> int\n\nRetuns structure size" },
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
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "ConfigV1", module );
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

  return pypdsdata::EvrData::PulseConfig::PyObject_FromPds( obj->pulse(idx) );
}

PyObject*
output_map( PyObject* self, PyObject* args )
{
  const Pds::EvrData::ConfigV1* obj = pypdsdata::EvrData::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:EvrData.ConfigV1.output_map", &idx ) ) return 0;

  return pypdsdata::EvrData::OutputMap::PyObject_FromPds( obj->output_map(idx) );
}

PyObject*
_repr( PyObject *self )
{
  Pds::EvrData::ConfigV1* obj = pypdsdata::EvrData::ConfigV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "evr.ConfigV1("; 

  str << "pulses=["; 
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
