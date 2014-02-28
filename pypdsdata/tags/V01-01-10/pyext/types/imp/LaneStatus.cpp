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
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Imp::LaneStatus, linkErrCount)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Imp::LaneStatus, linkDownCount)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Imp::LaneStatus, cellErrCount)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Imp::LaneStatus, rxCount)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Imp::LaneStatus, locLinked)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Imp::LaneStatus, remLinked)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Imp::LaneStatus, zeros)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Imp::LaneStatus, powersOkay)

  PyMethodDef methods[] = {
    { "linkErrCount",     linkErrCount,   METH_NOARGS, "self.linkErrCount() -> int\n\nReturns integer number." },
    { "linkDownCount",    linkDownCount,  METH_NOARGS, "self.linkDownCount() -> int\n\nReturns integer number." },
    { "cellErrCount",     cellErrCount,   METH_NOARGS, "self.cellErrCount() -> int\n\nReturns integer number." },
    { "rxCount",          rxCount,        METH_NOARGS, "self.rxCount() -> int\n\nReturns integer number." },
    { "locLinked",        locLinked,      METH_NOARGS, "self.locLinked() -> int\n\nReturns integer number." },
    { "remLinked",        remLinked,      METH_NOARGS, "self.remLinked() -> int\n\nReturns integer number." },
    { "zeros",            zeros,          METH_NOARGS, "self.zeros() -> int\n\nReturns integer number." },
    { "powersOkay",       powersOkay,     METH_NOARGS, "self.powersOkay() -> int\n\nReturns integer number." },
    {0, 0, 0, 0}
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
  type->tp_methods = ::methods;

  BaseType::initType( "LaneStatus", module );
}

void
pypdsdata::Imp::LaneStatus::print(std::ostream& out) const
{
  out << "imp.LaneStatus(linkErrCount=" << int(m_obj.linkErrCount())
      << ", linkDownCount=" << int(m_obj.linkDownCount())
      << ", cellErrCount=" << int(m_obj.cellErrCount())
      << ", rxCount=" << int(m_obj.rxCount())
      << ", locLinked=" << int(m_obj.locLinked())
      << ", remLinked=" << int(m_obj.remLinked())
      << ", zeros=" << int(m_obj.zeros())
      << ", powersOkay=" << int(m_obj.powersOkay())
      << ")";
}
