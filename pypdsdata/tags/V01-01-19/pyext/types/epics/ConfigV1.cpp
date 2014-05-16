//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV1...
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
#include "PvConfigV1.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::Epics::ConfigV1, numPv)
  FUN0_WRAPPER(pypdsdata::Epics::ConfigV1, getPvConfig)

  PyMethodDef methods[] = {
    {"numPv",       numPv,        METH_NOARGS,  "self.numPv() -> int\n\nReturns number of :py:class:`PvConfigV1` object." },
    {"getPvConfig", getPvConfig,  METH_NOARGS,  "self.getPvConfig() -> list of PvConfigV1\n\nReturns list of :py:class:`PvConfigV1` objects." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Epics::ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Epics::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::Epics::ConfigV1::print(std::ostream& str) const
{
  const ndarray<const Pds::Epics::PvConfigV1, 1>& pvConfigs = m_obj->getPvConfig();
  str << "epics.ConfigV1(numPv=" << pvConfigs.size() << ", PvConfigs=[";
  for ( unsigned i = 0; i < pvConfigs.size() and i < 256; ++ i ) {
    const Pds::Epics::PvConfigV1& pv = pvConfigs[i];
    if (i) str << ", ";
    str << "(pvId=" << pv.pvId() << ", desc=\"" << pv.description()
        << "\", interval=" << pv.interval() << ")";
  }
  if (pvConfigs.size() > 256) str << ", ...";
  str << "])";
}
