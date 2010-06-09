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
    { "numFifoEvents", numFifoEvents, METH_NOARGS,  "number of FIFOEvent objects" },
    { "fifoEvent",     fifoEvent,     METH_VARARGS, "" },
    { "size",          size,          METH_NOARGS,  "full size of the object" },
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
