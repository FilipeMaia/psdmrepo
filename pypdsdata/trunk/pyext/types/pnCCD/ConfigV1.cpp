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
  FUN0_WRAPPER(pypdsdata::PNCCD::ConfigV1, numLinks)
  FUN0_WRAPPER(pypdsdata::PNCCD::ConfigV1, payloadSizePerLink)

  PyMethodDef methods[] = {
    {"numLinks",           numLinks,           METH_NOARGS,  "self.numLinks() -> int\n\nReturns number of links." },
    {"payloadSizePerLink", payloadSizePerLink, METH_NOARGS,  "self.payloadSizePerLink() -> int\n\nReturns data size per link." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::PNCCD::ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::PNCCD::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::PNCCD::ConfigV1::print(std::ostream& str) const
{
  str << "pnccd.ConfigV1(numLinks=" << m_obj->numLinks() << ", payloadSizePerLink=" << m_obj->payloadSizePerLink() << ")";
}
