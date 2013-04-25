//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_DataV3...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataV3.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "DataV3_FIFOEvent.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::EvrData::DataV3, numFifoEvents)
  FUN0_WRAPPER(pypdsdata::EvrData::DataV3, size)
  PyObject* fifoEvent( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    { "numFifoEvents", numFifoEvents, METH_NOARGS,  "self.numFifoEvents() -> int\n\nReturns number of :py:class:`DataV3_FIFOEvent` objects" },
    { "fifoEvent",     fifoEvent,     METH_VARARGS, "self.fifoEvent(i: int) -> DataV3_FIFOEvent\n\nReturns :py:class:`DataV3_FIFOEvent` object" },
    { "size",          size,          METH_NOARGS,  "self.size() ->int\n\nReturns full size of the data object" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::DataV3 class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::DataV3::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "DataV3", module );
}

void
pypdsdata::EvrData::DataV3::print(std::ostream& str) const
{
  str << "evr.DataV3(";

  str << "fifoEvents=[";
  for (unsigned i = 0; i != m_obj->numFifoEvents(); ++ i ) {
    if (i != 0) str << ", ";
    const Pds::EvrData::DataV3::FIFOEvent& ev = m_obj->fifoEvent(i);
    str << ev.EventCode << ':' << ev.TimestampHigh << '.' << ev.TimestampLow;
  }
  str << "]";

  str << ")";
}

namespace {

PyObject*
fifoEvent( PyObject* self, PyObject* args )
{
  const Pds::EvrData::DataV3* obj = pypdsdata::EvrData::DataV3::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:EvrData.DataV3.eventcode", &idx ) ) return 0;

  return pypdsdata::EvrData::DataV3_FIFOEvent::PyObject_FromPds( obj->fifoEvent(idx) );
}

}
