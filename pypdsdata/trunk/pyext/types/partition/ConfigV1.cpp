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
#include "Source.h"
#include "../TypeLib.h"
#include "../../Src.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::Partition::ConfigV1, bldMask)
  FUN0_WRAPPER(pypdsdata::Partition::ConfigV1, numSources)
  FUN0_WRAPPER(pypdsdata::Partition::ConfigV1, sources)

  PyMethodDef methods[] = {
    {"bldMask",     bldMask,       METH_NOARGS,  "self.bldMask() -> int\n\nReturns BLD mask." },
    {"numSources",  numSources,    METH_NOARGS,  "self.numSources() -> int\n\nReturns number of source." },
    {"sources",     sources,       METH_NOARGS,  "self.sources() -> list\n\nReturns list of :py:class:`Source` objects." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Partition::ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Partition::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::Partition::ConfigV1::print(std::ostream& str) const
{
  str << "Partition.ConfigV1(bldMask=" << std::hex << std::showbase << m_obj->bldMask() << std::dec
      << ", numSources=" << m_obj->numSources()
      << ", sources=[";
  const ndarray<const Pds::Partition::Source, 1>& sources = m_obj->sources();
  for (unsigned i = 0; i != sources.size(); ++ i) {
    const Pds::Partition::Source& src = sources[i];
    if (i != 0) str << ", ";
    Src::print(str, src.src());
    str << ": " << src.group();
    
  }
  str << "])";
}
