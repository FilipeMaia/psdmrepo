//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PvConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PvConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "epicsTimeStamp.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  MEMBER_WRAPPER(pypdsdata::Epics::PvConfigV1, iPvId)
  MEMBER_WRAPPER(pypdsdata::Epics::PvConfigV1, sPvDesc)
  MEMBER_WRAPPER(pypdsdata::Epics::PvConfigV1, fInterval)

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"iPvId",       iPvId,          0, "Integer number, PV Id", 0},
    {"sPvDesc",     sPvDesc,        0, "String, PV description", 0},
    {"fInterval",   fInterval,      0, "Floating number", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Epics::PvConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Epics::PvConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "PvConfigV1", module );
}

void
pypdsdata::Epics::PvConfigV1::print(std::ostream& str) const
{
  str << "PvConfigV1(iPvId=" << m_obj->iPvId
      << ", sPvDesc=" << m_obj->sPvDesc
      << ", fInterval=" << m_obj->fInterval
      << ")";
}
