//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ElementV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ElementV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Sample.h"
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::Imp::ElementV1, vc)
  FUN0_WRAPPER(pypdsdata::Imp::ElementV1, lane)
  FUN0_WRAPPER(pypdsdata::Imp::ElementV1, frameNumber)
  FUN0_WRAPPER(pypdsdata::Imp::ElementV1, range)
  FUN0_WRAPPER(pypdsdata::Imp::ElementV1, laneStatus)
  PyObject* sample( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    { "vc",             vc,             METH_NOARGS,  "self.vc() -> int\n\nReturns integer number." },
    { "lane",           lane,           METH_NOARGS,  "self.lane() -> int\n\nReturns integer number." },
    { "frameNumber",    frameNumber,    METH_NOARGS,
        "self.frameNumber() -> int\n\nReturns frame number." },
    { "range",          range,          METH_NOARGS,  "self.range() -> int\n\nReturns integer number." },
    { "laneStatus",     laneStatus,     METH_NOARGS,  "self.laneStatus() -> int\n\nReturns integer number." },
    { "sample",         sample,         METH_VARARGS,
        "self.sample(index:int) -> object\n\nReturns instance of :py:class:`Sample` class for given sample index." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Imp::ElementV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Imp::ElementV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ElementV1", module );
}

void
pypdsdata::Imp::ElementV1::print(std::ostream& out) const
{
  out << "imp.ElementV1(frameNumber=" << m_obj->frameNumber() << ", samples=[[";
  Pds::Imp::Sample s = m_obj->getSample(0);
  for (unsigned i = 0; i != Pds::Imp::channelsPerDevice; ++ i) {
    if (i > 0) out << ", ";
    out << s.channel(i);
  }
  out << "], ...])";
}

namespace {

PyObject*
sample( PyObject* self, PyObject* args )
{
  Pds::Imp::ElementV1* obj = pypdsdata::Imp::ElementV1::pdsObject(self);
  if (not obj) return 0;

  unsigned index;
  if (not PyArg_ParseTuple(args, "I:imp.ElementV1.sample", &index)) return 0;

  return pypdsdata::Imp::Sample::PyObject_FromPds(obj->getSample(index));
}

}
