//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_ConfigV1...
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
#include "../TypeLib.h"
#include "HorizV1.h"
#include "TrigV1.h"
#include "VertV1.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::Acqiris::ConfigV1, nbrConvertersPerChannel)
  FUN0_WRAPPER(pypdsdata::Acqiris::ConfigV1, channelMask)
  FUN0_WRAPPER(pypdsdata::Acqiris::ConfigV1, nbrChannels)
  FUN0_WRAPPER(pypdsdata::Acqiris::ConfigV1, nbrBanks)
  PyObject* horiz( PyObject* self, PyObject* );
  PyObject* trig( PyObject* self, PyObject* );
  PyObject* vert( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"nbrConvertersPerChannel", nbrConvertersPerChannel, METH_NOARGS,  "Returns integer number" },
    {"channelMask",  channelMask,  METH_NOARGS,  "Returns integer number" },
    {"nbrChannels",  nbrChannels,  METH_NOARGS,  "Returns integer number" },
    {"nbrBanks",     nbrBanks,     METH_NOARGS,  "Returns integer number" },
    {"horiz",        horiz,        METH_NOARGS,  "Returns HorizV1 object" },
    {"trig",         trig,         METH_NOARGS,  "Returns TrigV1 object" },
    {"vert",         vert,         METH_VARARGS, "Returns VertV1 object for a given channel number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV1", module );
}

void 
pypdsdata::Acqiris::ConfigV1::print(std::ostream& out) const
{
  if(not m_obj) {
    out << "acqiris.ConfigV1(None)";
  } else {  
    out << "acqiris.ConfigV1(nCPC=" << m_obj->nbrConvertersPerChannel()
        << ", chMask=" << m_obj->channelMask()
        << ", nCh=" << m_obj->nbrChannels()
        << ", nBanks=" << m_obj->nbrBanks()
        << ", ...)" ;
  }
}

namespace {

PyObject*
horiz( PyObject* self, PyObject* )
{
  const Pds::Acqiris::ConfigV1* obj = pypdsdata::Acqiris::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::Acqiris::HorizV1::PyObject_FromPds( (Pds::Acqiris::HorizV1*)&obj->horiz(), self, sizeof(Pds::Acqiris::HorizV1) );
}

PyObject*
trig( PyObject* self, PyObject* )
{
  const Pds::Acqiris::ConfigV1* obj = pypdsdata::Acqiris::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::Acqiris::TrigV1::PyObject_FromPds( (Pds::Acqiris::TrigV1*)&obj->trig(), self, sizeof(Pds::Acqiris::TrigV1) );
}


PyObject*
vert( PyObject* self, PyObject* args )
{
  const Pds::Acqiris::ConfigV1* obj = pypdsdata::Acqiris::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  unsigned channel;
  if ( not PyArg_ParseTuple( args, "I:Acqiris.ConfigV1.vert", &channel ) ) return 0;

  return pypdsdata::Acqiris::VertV1::PyObject_FromPds( (Pds::Acqiris::VertV1*)&obj->vert(channel), self, sizeof(Pds::Acqiris::VertV1) );
}

}
