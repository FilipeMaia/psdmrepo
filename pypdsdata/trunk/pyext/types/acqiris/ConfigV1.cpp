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
  FUN0_WRAPPER(pypdsdata::Acqiris::ConfigV1, horiz)
  FUN0_WRAPPER(pypdsdata::Acqiris::ConfigV1, trig)
  PyObject* vert( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"nbrConvertersPerChannel", nbrConvertersPerChannel, METH_NOARGS,  "self.nbrConvertersPerChannel() -> int\n\nReturns integer number" },
    {"channelMask",  channelMask,  METH_NOARGS,  "self.channelMask() -> int\n\nReturns integer number" },
    {"nbrChannels",  nbrChannels,  METH_NOARGS,  "self.nbrChannels() -> int\n\nReturns integer number" },
    {"nbrBanks",     nbrBanks,     METH_NOARGS,  "self.nbrBanks() -> int\n\nReturns integer number" },
    {"horiz",        horiz,        METH_NOARGS,  "self.horiz() -> HorizV1\n\nReturns :py:class:`HorizV1` object" },
    {"trig",         trig,         METH_NOARGS,  "self.trig() -> TrigV1\n\nReturns :py:class:`TrigV1` object" },
    {"vert",         vert,         METH_VARARGS,
        "self.vert([channel: int]) -> VertV1\n\nWithout argument returns the list of :py:class:`VertV1` objects, "
        "if argument is given then returns single object for a given channel number." },
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
        << ", horiz(nSeg=" << m_obj->horiz().nbrSegments()
        << ", nSampl=" << m_obj->horiz().nbrSamples()
        << "))";
  }
}

namespace {

PyObject*
vert( PyObject* self, PyObject* args )
{
  const Pds::Acqiris::ConfigV1* obj = pypdsdata::Acqiris::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  unsigned channel = unsigned(-1);
  if ( not PyArg_ParseTuple( args, "|I:Acqiris.ConfigV1.vert", &channel ) ) return 0;

  const ndarray<const Pds::Acqiris::VertV1, 1>& vert = obj->vert();

  // if argument is missing the return list of objects, otherwise return single object
  using pypdsdata::TypeLib::toPython;
  if (channel == unsigned(-1)) {
    return toPython(vert);
  } else {
    return toPython(vert[channel]);
  }
}

}
