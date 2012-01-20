//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TdcConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"
#include "TdcChannel.h"
#include "TdcAuxIO.h"
#include "TdcVetoIO.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  PyObject* channel( PyObject* self, PyObject* args );
  PyObject* auxio( PyObject* self, PyObject* args );
  PyObject* veto( PyObject* self, PyObject* );

  PyMethodDef methods[] = {
    {"channel", channel, METH_VARARGS, "Returns TdcChannel object for a given index" },
    {"auxio",   auxio,   METH_VARARGS, "Returns TdcAuxIO object for a given index" },
    {"veto",    veto,    METH_NOARGS,  "Returns TdcVetoIO object" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::TdcConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::TdcConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums and constants
  PyObject* tp_dict = PyDict_New();
  PyObject* val = PyInt_FromLong(Pds::Acqiris::TdcConfigV1::NChannels);
  PyDict_SetItemString( tp_dict, "NChannels", val );
  val = PyInt_FromLong(Pds::Acqiris::TdcConfigV1::NAuxIO);
  PyDict_SetItemString( tp_dict, "NAuxIO", val );
  type->tp_dict = tp_dict;

  BaseType::initType( "TdcConfigV1", module );
}

void 
pypdsdata::Acqiris::TdcConfigV1::print(std::ostream& out) const
{
  if(not m_obj) return;

  out << "acqiris.TdcConfigV1(...)" ;
}

namespace {

PyObject*
channel( PyObject* self, PyObject* args )
{
  const Pds::Acqiris::TdcConfigV1* obj = pypdsdata::Acqiris::TdcConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  unsigned channel;
  if ( not PyArg_ParseTuple( args, "I:Acqiris.TdcConfigV1.channel", &channel ) ) return 0;

  return pypdsdata::Acqiris::TdcChannel::PyObject_FromPds( 
      (Pds::Acqiris::TdcChannel*)&obj->channel(channel), self, sizeof(Pds::Acqiris::TdcChannel) );
}

PyObject*
auxio( PyObject* self, PyObject* args )
{
  const Pds::Acqiris::TdcConfigV1* obj = pypdsdata::Acqiris::TdcConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  unsigned auxio;
  if ( not PyArg_ParseTuple( args, "I:Acqiris.TdcConfigV1.auxio", &auxio ) ) return 0;

  return pypdsdata::Acqiris::TdcAuxIO::PyObject_FromPds( 
      (Pds::Acqiris::TdcAuxIO*)&obj->auxio(auxio), self, sizeof(Pds::Acqiris::TdcAuxIO) );
}


PyObject*
veto( PyObject* self, PyObject* )
{
  const Pds::Acqiris::TdcConfigV1* obj = pypdsdata::Acqiris::TdcConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::Acqiris::TdcVetoIO::PyObject_FromPds( 
      (Pds::Acqiris::TdcVetoIO*)&obj->veto(), self, sizeof(Pds::Acqiris::TdcVetoIO) );
}

}
