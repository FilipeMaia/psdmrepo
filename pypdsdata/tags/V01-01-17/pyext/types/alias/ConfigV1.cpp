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
#include "SrcAlias.h"
#include "../TypeLib.h"
#include "../../Src.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::Alias::ConfigV1, numSrcAlias)
  FUN0_WRAPPER(pypdsdata::Alias::ConfigV1, srcAlias)

  PyMethodDef methods[] = {
    {"numSrcAlias",     numSrcAlias,       METH_NOARGS,  "self.numSrcAlias() -> int\n\nReturns number of alias definitions." },
    {"srcAlias",        srcAlias,          METH_NOARGS,  "self.srcAlias() -> list\n\nReturns list of :py:class:`SrcAlias` objects." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Alias::ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Alias::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::Alias::ConfigV1::print(std::ostream& str) const
{
  str << "Alias.ConfigV1(numSrcAlias=" << m_obj->numSrcAlias()
      << ", aliases=[";
  const ndarray<const Pds::Alias::SrcAlias, 1>& aliases = m_obj->srcAlias();
  for (unsigned i = 0; i != aliases.size(); ++ i) {
    const Pds::Alias::SrcAlias& alias = aliases[i];
    if (i != 0) str << ", ";
    Src::print(str, alias.src());
    str << ": \"" << alias.aliasName() << '"';
    
  }
  str << "])";
}
