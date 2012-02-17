//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_SequencerConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "SequencerConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../EnumType.h"
#include "SequencerEntry.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  pypdsdata::EnumType::Enum sourceEnumValues[] = {
      { "r120Hz",    Pds::EvrData::SequencerConfigV1::r120Hz },
      { "r60Hz",     Pds::EvrData::SequencerConfigV1::r60Hz },
      { "r30Hz",     Pds::EvrData::SequencerConfigV1::r30Hz },
      { "r10Hz",     Pds::EvrData::SequencerConfigV1::r10Hz },
      { "r5Hz",      Pds::EvrData::SequencerConfigV1::r5Hz },
      { "r1Hz",      Pds::EvrData::SequencerConfigV1::r1Hz },
      { "r0_5Hz",    Pds::EvrData::SequencerConfigV1::r0_5Hz },
      { "Disable",   Pds::EvrData::SequencerConfigV1::Disable },
      { 0, 0 }
  };
  pypdsdata::EnumType sourceEnum ( "Source", sourceEnumValues );

  // type-specific methods
  ENUM_FUN0_WRAPPER(pypdsdata::EvrData::SequencerConfigV1, sync_source, sourceEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::EvrData::SequencerConfigV1, beam_source, sourceEnum)
  FUN0_WRAPPER(pypdsdata::EvrData::SequencerConfigV1, length)
  FUN0_WRAPPER(pypdsdata::EvrData::SequencerConfigV1, cycles)
  PyObject* entry( PyObject* self, PyObject* args );
  PyObject* entries( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    { "sync_source",   sync_source,  METH_NOARGS, "self.sync_source() -> Source enum\n\nReturns Source enum" },
    { "beam_source",   beam_source,  METH_NOARGS, "self.beam_source() -> Source enum\n\nReturns Source enum" },
    { "length",        length,       METH_NOARGS, "self.length() -> int\n\nReturns number of entries" },
    { "cycles",        cycles,       METH_NOARGS, "self.cycles() -> int\n\nReturns integer number" },
    { "entries",       entries,      METH_NOARGS, "self.entries() -> list\n\nReturns list of SequencerEntry objects" },
    { "entry",         entry,        METH_VARARGS, "self.entry(i: int) -> SequencerEntry\n\nReturns SequencerEntry object" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::SequencerConfigV1 class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::SequencerConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "SequencerConfigV1", module );
}

namespace {

PyObject*
entry( PyObject* self, PyObject* args )
{
  const Pds::EvrData::SequencerConfigV1* obj = pypdsdata::EvrData::SequencerConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  // get argument as index
  unsigned idx;
  if ( not PyArg_ParseTuple( args, "I:evr.SequencerConfigV1.entry", &idx ) ) return 0;

  return pypdsdata::EvrData::SequencerEntry::PyObject_FromPds( obj->entry(idx) );
}

PyObject*
entries( PyObject* self, PyObject* )
{
  const Pds::EvrData::SequencerConfigV1* obj = pypdsdata::EvrData::SequencerConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  // copy all entries to the Python list
  unsigned len = obj->length();
  PyObject* list = PyList_New( len );
  for (unsigned i = 0 ; i != len ; ++ i ) {
    PyObject* q = pypdsdata::EvrData::SequencerEntry::PyObject_FromPds( obj->entry(i) );
    PyList_SET_ITEM( list, i, q );
  }

  return list;
}

}
