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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../camera/FrameCoord.h"

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
    {"numLinks",                numLinks,               METH_NOARGS,  "Returns number of links." },
    {"payloadSizePerLink",      payloadSizePerLink,     METH_NOARGS,  "Returns data size per link in bytes." },
    {"numChannels",             numChannels,            METH_NOARGS,  "" },
    {"numRows",                 numRows,                METH_NOARGS,  "" },
    {"numSubmoduleChannels",    numSubmoduleChannels,   METH_NOARGS,  "" },
    {"numSubmoduleRows",        numSubmoduleRows,       METH_NOARGS,  "" },
    {"numSubmodules",           numSubmodules,          METH_NOARGS,  "" },
    {"camexMagic",              camexMagic,             METH_NOARGS,  "" },
    {"info",                    info,                   METH_NOARGS,  "" },
    {"timingFName",             timingFName,            METH_NOARGS,  "" },
    {"size",                    size,                   METH_NOARGS,  "" },
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
