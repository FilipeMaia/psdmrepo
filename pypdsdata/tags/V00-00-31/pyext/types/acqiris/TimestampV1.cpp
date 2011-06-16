//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TimestampV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TimestampV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

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
  FUN0_WRAPPER(pypdsdata::Acqiris::TimestampV1, pos)
  FUN0_WRAPPER(pypdsdata::Acqiris::TimestampV1, value)

  PyMethodDef methods[] = {
    {"pos",     pos,    METH_NOARGS,  "Returns floating number" },
    {"value",   value,  METH_NOARGS,  "Returns integer number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::TimestampV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::TimestampV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "TimestampV1", module );
}

void 
pypdsdata::Acqiris::TimestampV1::print(std::ostream& out) const
{
  if(not m_obj) {
    out << "acqiris.TimestampV1(None)";
  } else {  
    out << "acqiris.TimestampV1(" << m_obj->value() << ")" ;
  }
}
