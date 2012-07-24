//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataEBeamV3...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataEBeamV3.h"

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
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, uDamageMask)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamCharge)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamL3Energy)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamLTUPosX)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamLTUPosY)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamLTUAngX)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamLTUAngY)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamPkCurrBC2)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamEnergyBC2)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamPkCurrBC1)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamEnergyBC1)
  PyObject* _repr( PyObject *self );

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"uDamageMask",    uDamageMask,    0, "integer number", 0},
    {"fEbeamCharge",   fEbeamCharge,   0, "floating number, in nC", 0},
    {"fEbeamL3Energy", fEbeamL3Energy, 0, "floating number, in MeV", 0},
    {"fEbeamLTUPosX",  fEbeamLTUPosX,  0, "floating number, in mm", 0},
    {"fEbeamLTUPosY",  fEbeamLTUPosY,  0, "floating number, in mm", 0},
    {"fEbeamLTUAngX",  fEbeamLTUAngX,  0, "floating number, in mrad", 0},
    {"fEbeamLTUAngY",  fEbeamLTUAngY,  0, "floating number, in mrad", 0},
    {"fEbeamPkCurrBC2", fEbeamPkCurrBC2, 0, "floating number, in Amps", 0},
    {"fEbeamEnergyBC2", fEbeamEnergyBC2, 0, "floating number, beam position (related to energy), in mm", 0},
    {"fEbeamPkCurrBC1", fEbeamPkCurrBC1, 0, "floating number, in Amps", 0},
    {"fEbeamEnergyBC1", fEbeamEnergyBC1, 0, "floating number, beam position (related to energy), in mm", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::BldDataEBeamV3 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataEBeamV3::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "BldDataEBeamV3", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::BldDataEBeamV3* pdsObj = pypdsdata::BldDataEBeamV3::pdsObject(self);
  if(not pdsObj) return 0;

  char buf[64];
  snprintf( buf, sizeof buf, "BldDataEBeamV3(Charge=%g, L3Energy=%g, ...)",
            pdsObj->fEbeamCharge, pdsObj->fEbeamL3Energy );
  return PyString_FromString( buf );
}

}
