//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_IOChannel...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "IOChannel.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../../DetInfo.h"
#include "OutputMap.h"
#include "EventCodeV4.h"
#include "PulseConfigV3.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::EvrData::IOChannel, name)
  FUN0_WRAPPER(pypdsdata::EvrData::IOChannel, ninfo)
  PyObject* info( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    { "name",    name,   METH_NOARGS,  "self.name() -> string\n\nReturns string" },
    { "ninfo",   ninfo,  METH_NOARGS,  "self.ninfo() -> int\n\nReturns number of DetInfo objects" },
    { "info",    info,   METH_VARARGS, "self.info(i: int) -> xtc.DetInfo\n\nReturns DetInfo object" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::IOChannel class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::IOChannel::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "IOChannel", module );
}

namespace {

PyObject*
info( PyObject* self, PyObject* args )
{
  const Pds::EvrData::IOChannel* obj = pypdsdata::EvrData::IOChannel::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:EvrData.IOChannel.info", &idx ) ) return 0;

  return pypdsdata::DetInfo::PyObject_FromPds( obj->info(idx) );
}

}
