//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_IOConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "IOConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../../EnumType.h"
#include "IOChannel.h"
#include "OutputMap.h"
#include "EventCodeV4.h"
#include "PulseConfigV3.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  ENUM_FUN0_WRAPPER(pypdsdata::EvrData::IOConfigV1, conn, pypdsdata::EvrData::OutputMap::connEnum())
  FUN0_WRAPPER(pypdsdata::EvrData::IOConfigV1, nchannels)
  PyObject* channel( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    { "conn",      conn,       METH_NOARGS,  "self.conn() -> OutputMap.Conn enum\n\nReturns :py:class:`OutputMap.Conn` enum" },
    { "nchannels", nchannels,  METH_NOARGS,  "self.nchannels() -> int\n\nReturns number of channels" },
    { "channel",   channel,    METH_VARARGS, "self.channel(i: int) -> IOChannel\n\nReturns :py:class:`IOChannel` object" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::IOConfigV1 class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::IOConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "IOConfigV1", module );
}

namespace {

PyObject*
channel( PyObject* self, PyObject* args )
{
  const Pds::EvrData::IOConfigV1* obj = pypdsdata::EvrData::IOConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:EvrData.IOConfigV1.channel", &idx ) ) return 0;

  Pds::EvrData::IOChannel& chan = const_cast<Pds::EvrData::IOChannel&>(obj->channel(idx));
  return pypdsdata::EvrData::IOChannel::PyObject_FromPds( &chan, self, chan.size() );
}

}
