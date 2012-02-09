//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataPhaseCavity...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataPhaseCavity.h"

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
  MEMBER_WRAPPER(pypdsdata::BldDataPhaseCavity, fFitTime1)
  MEMBER_WRAPPER(pypdsdata::BldDataPhaseCavity, fFitTime2)
  MEMBER_WRAPPER(pypdsdata::BldDataPhaseCavity, fCharge1)
  MEMBER_WRAPPER(pypdsdata::BldDataPhaseCavity, fCharge2)
  PyObject* _repr( PyObject *self );

  PyGetSetDef getset[] = {
    {"fFitTime1", fFitTime1, 0, "floating number, PV name: UND:R02:IOC:16:BAT:FitTime1, in pico-seconds", 0},
    {"fFitTime2", fFitTime2, 0, "floating number, PV name: UND:R02:IOC:16:BAT:FitTime2, in pico-seconds", 0},
    {"fCharge1",  fCharge1,  0, "floating number, PV name: UND:R02:IOC:16:BAT:Charge1, in pico-columbs", 0},
    {"fCharge2",  fCharge2,  0, "floating number, PV name: UND:R02:IOC:16:BAT:Charge2, in pico-columbs", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::BldDataPhaseCavity class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataPhaseCavity::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "BldDataPhaseCavity", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::BldDataPhaseCavity* pdsObj = pypdsdata::BldDataPhaseCavity::pdsObject(self);
  if(not pdsObj) return 0;

  char buf[96];
  snprintf( buf, sizeof buf, "BldDataPhaseCavity(ft1=%f, ft2=%f, ch1=%f, ch2=%f)",
            pdsObj->fFitTime1, pdsObj->fFitTime2, pdsObj->fCharge1, pdsObj->fCharge2 );
  return PyString_FromString( buf );
}

}
