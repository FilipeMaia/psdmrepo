//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataFEEGasDetEnergy...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataFEEGasDetEnergy.h"

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
  MEMBER_WRAPPER(pypdsdata::BldDataFEEGasDetEnergy, f_11_ENRC)
  MEMBER_WRAPPER(pypdsdata::BldDataFEEGasDetEnergy, f_12_ENRC)
  MEMBER_WRAPPER(pypdsdata::BldDataFEEGasDetEnergy, f_21_ENRC)
  MEMBER_WRAPPER(pypdsdata::BldDataFEEGasDetEnergy, f_22_ENRC)
  PyObject* _repr( PyObject *self );

  PyGetSetDef getset[] = {
    {"f_11_ENRC",   f_11_ENRC,   0, "PV name: GDET:FEE1:11:ENRC", 0},
    {"f_12_ENRC",   f_12_ENRC,   0, "PV name: GDET:FEE1:12:ENRC", 0},
    {"f_21_ENRC",   f_21_ENRC,   0, "PV name: GDET:FEE1:21:ENRC", 0},
    {"f_22_ENRC",   f_22_ENRC,   0, "PV name: GDET:FEE1:22:ENRC", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::BldDataFEEGasDetEnergy class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataFEEGasDetEnergy::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "BldDataFEEGasDetEnergy", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::BldDataFEEGasDetEnergy* pdsObj = pypdsdata::BldDataFEEGasDetEnergy::pdsObject(self);
  if(not pdsObj) return 0;

  char buf[96];
  snprintf( buf, sizeof buf, "BldDataFEEGasDetEnergy(11=%f, 12=%f, 21=%f, 22=%f)",
            pdsObj->f_11_ENRC, pdsObj->f_12_ENRC, pdsObj->f_21_ENRC, pdsObj->f_22_ENRC );
  return PyString_FromString( buf );
}

}
