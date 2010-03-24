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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Exception.h"
#include "OutputMap.h"
#include "PulseConfig.h"
#include "types/TypeLib.h"

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

  PyMethodDef methods[] = {
    { "npulses",    npulses,     METH_NOARGS, "pulse configurations appended to this structure" },
    { "pulse",      pulse,       METH_NOARGS, "pulse configurations appended to this structure" },
    { "noutputs",   noutputs,    METH_NOARGS, "output configurations appended to this structure" },
    { "output_map", output_map,  METH_NOARGS, "output configurations appended to this structure" },
    { "size",       size,        METH_NOARGS, "size including appended PulseConfig's and OutputMap's" },
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

}
