//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataEBeam...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataEBeam.h"

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
  MEMBER_WRAPPER(pypdsdata::BldDataEBeam, uDamageMask)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeam, fEbeamCharge)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeam, fEbeamL3Energy)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeam, fEbeamLTUPosX)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeam, fEbeamLTUPosY)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeam, fEbeamLTUAngX)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeam, fEbeamLTUAngY)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeam, fEbeamPkCurrBC2)
  PyObject* _repr( PyObject *self );

  PyGetSetDef getset[] = {
    {"uDamageMask",    uDamageMask,    0, "", 0},
    {"fEbeamCharge",   fEbeamCharge,   0, "in nC", 0},
    {"fEbeamL3Energy", fEbeamL3Energy, 0, "in MeV", 0},
    {"fEbeamLTUPosX",  fEbeamLTUPosX,  0, "in mm", 0},
    {"fEbeamLTUPosY",  fEbeamLTUPosY,  0, "in mm", 0},
    {"fEbeamLTUAngX",  fEbeamLTUAngX,  0, "in mrad", 0},
    {"fEbeamLTUAngY",  fEbeamLTUAngY,  0, "in mrad", 0},
    {"fEbeamPkCurrBC2", fEbeamPkCurrBC2, 0, "in Amps", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::BldDataEBeam class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataEBeam::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "BldDataEBeam", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::BldDataEBeam* pdsObj = pypdsdata::BldDataEBeam::pdsObject(self);
  if(not pdsObj) return 0;

  char buf[64];
  snprintf( buf, sizeof buf, "BldDataEBeam(Charge=%f, L3Energy=%f, ...)",
            pdsObj->fEbeamCharge, pdsObj->fEbeamL3Energy );
  return PyString_FromString( buf );
}

}
