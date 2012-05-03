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

  // standard Python stuff
  PyObject* _repr( PyObject *self );

  // methods
  MEMBER_WRAPPER(pypdsdata::Epics::PvConfigV1, iPvId)
  MEMBER_WRAPPER(pypdsdata::Epics::PvConfigV1, sPvDesc)
  MEMBER_WRAPPER(pypdsdata::Epics::PvConfigV1, fInterval)

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
  type->tp_str = _repr ;
  type->tp_repr = _repr ;

  BaseType::initType( "PvConfigV1", module );
}


namespace {


PyObject*
_repr( PyObject *self )
{
  Pds::Epics::PvConfigV1* obj = pypdsdata::Epics::PvConfigV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "PvConfigV1(iPvId=" << obj->iPvId
      << ", sPvDesc=" << obj->sPvDesc
      << ", fInterval=" << obj->fInterval
      << ")";
  
  return PyString_FromString(str.str().c_str());
}

}
