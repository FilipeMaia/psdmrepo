//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV2.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::PNCCD::ConfigV2, numLinks)
  FUN0_WRAPPER(pypdsdata::PNCCD::ConfigV2, payloadSizePerLink)
  FUN0_WRAPPER(pypdsdata::PNCCD::ConfigV2, numChannels)
  FUN0_WRAPPER(pypdsdata::PNCCD::ConfigV2, numRows)
  FUN0_WRAPPER(pypdsdata::PNCCD::ConfigV2, numSubmoduleChannels)
  FUN0_WRAPPER(pypdsdata::PNCCD::ConfigV2, numSubmoduleRows)
  FUN0_WRAPPER(pypdsdata::PNCCD::ConfigV2, numSubmodules)
  FUN0_WRAPPER(pypdsdata::PNCCD::ConfigV2, camexMagic)
  FUN0_WRAPPER(pypdsdata::PNCCD::ConfigV2, info)
  FUN0_WRAPPER(pypdsdata::PNCCD::ConfigV2, timingFName)
  FUN0_WRAPPER(pypdsdata::PNCCD::ConfigV2, size)

  PyMethodDef methods[] = {
    {"numLinks",                numLinks,               METH_NOARGS,  "self.numLinks() -> int\n\nReturns number of links." },
    {"payloadSizePerLink",      payloadSizePerLink,     METH_NOARGS,  "self.payloadSizePerLink() -> int\n\nReturns data size per link in bytes." },
    {"numChannels",             numChannels,            METH_NOARGS,  "self.numChannels() -> int\n\nReturns integer number" },
    {"numRows",                 numRows,                METH_NOARGS,  "self.numRows() -> int\n\nReturns integer number" },
    {"numSubmoduleChannels",    numSubmoduleChannels,   METH_NOARGS,  "self.numSubmoduleChannels() -> int\n\nReturns integer number" },
    {"numSubmoduleRows",        numSubmoduleRows,       METH_NOARGS,  "self.numSubmoduleRows() -> int\n\nReturns integer number" },
    {"numSubmodules",           numSubmodules,          METH_NOARGS,  "self.numSubmodules() -> int\n\nReturns integer number" },
    {"camexMagic",              camexMagic,             METH_NOARGS,  "self.camexMagic() -> int\n\nReturns integer number" },
    {"info",                    info,                   METH_NOARGS,  "self.info() -> string\n\nReturns informational string" },
    {"timingFName",             timingFName,            METH_NOARGS,  "self.timingFName() -> string\n\nReturns timing file name" },
    {"size",                    size,                   METH_NOARGS,  "self.size() -> int\n\nReturns total size of the object" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::PNCCD::ConfigV2 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::PNCCD::ConfigV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV2", module );
}

void
pypdsdata::PNCCD::ConfigV2::print(std::ostream& str) const
{
  str << "pnccd.ConfigV2(numLinks=" << m_obj->numLinks()
      << ", payloadSizePerLink=" << m_obj->payloadSizePerLink()
      << ", numChannels=" << m_obj->numChannels()
      << ", numRows=" << m_obj->numRows()
      << ", ...)";
}
