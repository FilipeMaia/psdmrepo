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
#include "LaneStatus.h"
#include "Sample.h"
#include "ConfigV1.h"
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
  PyObject* samples( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    { "vc",             vc,             METH_NOARGS,  "self.vc() -> int\n\nReturns integer number." },
    { "lane",           lane,           METH_NOARGS,  "self.lane() -> int\n\nReturns integer number." },
    { "frameNumber",    frameNumber,    METH_NOARGS,  "self.frameNumber() -> int\n\nReturns frame number." },
    { "range",          range,          METH_NOARGS,  "self.range() -> int\n\nReturns integer number." },
    { "laneStatus",     laneStatus,     METH_NOARGS,  "self.laneStatus() -> int\n\nReturns integer number." },
    { "samples",        samples,        METH_NOARGS,
        "self.samples() -> list\n\nReturns list of :py:class:`Sample` objects." },
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
  out << "imp.ElementV1(frameNumber=" << m_obj->frameNumber()
      << ", range=" << m_obj->range()
      << ", vc=" << int(m_obj->vc())
      << ", lane=" << int(m_obj->lane())
      << ", ...)";
}

namespace {

PyObject*
samples( PyObject* self, PyObject* args )
{
  Pds::Imp::ElementV1* obj = pypdsdata::Imp::ElementV1::pdsObject(self);
  if (not obj) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:Imp.ElementV1.data", &configObj ) ) return 0;

  // get config object
  Pds::Imp::ConfigV1* config = 0;
  if ( pypdsdata::Imp::ConfigV1::Object_TypeCheck( configObj ) ) {
    config = pypdsdata::Imp::ConfigV1::pdsObject( configObj );
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a Imp.ConfigV* object");
    return 0;
  }

  return pypdsdata::TypeLib::toPython(obj->samples(*config));
}

}
