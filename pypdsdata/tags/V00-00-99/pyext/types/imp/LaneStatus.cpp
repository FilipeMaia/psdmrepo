//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LaneStatus...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "LaneStatus.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // type-specific methods
  PyObject* usLinkErrCount( PyObject* self, void* );
  PyObject* usLinkDownCount( PyObject* self, void* );
  PyObject* usCellErrCount( PyObject* self, void* );
  PyObject* usRxCount( PyObject* self, void* );
  PyObject* usLocLinked( PyObject* self, void* );
  PyObject* usRemLinked( PyObject* self, void* );
  PyObject* zeros( PyObject* self, void* );
  PyObject* powersOkay( PyObject* self, void* );

#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
      { "usLinkErrCount",    usLinkErrCount,  0, "Integer number." },
      { "usLinkDownCount",   usLinkDownCount, 0, "Integer number." },
      { "usCellErrCount",    usCellErrCount,  0, "Integer number." },
      { "usRxCount",         usRxCount,       0, "Integer number." },
      { "usLocLinked",       usLocLinked,     0, "Integer number." },
      { "usRemLinked",       usRemLinked,     0, "Integer number." },
      { "zeros",             zeros,           0, "Integer number." },
      { "powersOkay",        powersOkay,      0, "Integer number." },
      {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Imp::LaneStatus class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Imp::LaneStatus::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "LaneStatus", module );
}

void
pypdsdata::Imp::LaneStatus::print(std::ostream& out) const
{
  out << "imp.LaneStatus(usLinkErrCount=" << m_obj.usLinkErrCount
      << ", usLinkDownCount=" << usLinkDownCount
      << ")";
}

namespace {

PyObject*
usLinkErrCount( PyObject* self, void* )
{
  return PyInt_FromLong(pypdsdata::Imp::LaneStatus::pdsObject(self).usLinkErrCount);
}

PyObject*
usLinkDownCount( PyObject* self, void* )
{
  return PyInt_FromLong(pypdsdata::Imp::LaneStatus::pdsObject(self).usLinkDownCount);
}

PyObject*
usCellErrCount( PyObject* self, void* )
{
  return PyInt_FromLong(pypdsdata::Imp::LaneStatus::pdsObject(self).usCellErrCount);
}

PyObject*
usRxCount( PyObject* self, void* )
{
  return PyInt_FromLong(pypdsdata::Imp::LaneStatus::pdsObject(self).usRxCount);
}

PyObject*
usLocLinked( PyObject* self, void* )
{
  return PyInt_FromLong(pypdsdata::Imp::LaneStatus::pdsObject(self).usLocLinked);
}

PyObject*
usRemLinked( PyObject* self, void* )
{
  return PyInt_FromLong(pypdsdata::Imp::LaneStatus::pdsObject(self).usRemLinked);
}

PyObject*
zeros( PyObject* self, void* )
{
  return PyInt_FromLong(pypdsdata::Imp::LaneStatus::pdsObject(self).zeros);
}

PyObject*
powersOkay( PyObject* self, void* )
{
  return PyInt_FromLong(pypdsdata::Imp::LaneStatus::pdsObject(self).powersOkay);
}

}
