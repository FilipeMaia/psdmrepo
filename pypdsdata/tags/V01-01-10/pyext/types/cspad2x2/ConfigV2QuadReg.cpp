//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV2QuadReg...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV2QuadReg.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CsPad2x2DigitalPotsCfg.h"
#include "CsPad2x2GainMapCfg.h"
#include "CsPad2x2ReadOnlyCfg.h"
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  pypdsdata::EnumType::Enum dataModesEnumValues[] = {
      { "normal",     Pds::CsPad2x2::normal },
      { "shiftTest",  Pds::CsPad2x2::shiftTest },
      { "testData",   Pds::CsPad2x2::testData },
      { "reserved",   Pds::CsPad2x2::reserved },
      { 0, 0 }
  };
  pypdsdata::EnumType dataModesEnum ( "DataModes", dataModesEnumValues );

  // methods
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, shiftSelect)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, edgeSelect)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, readClkSet)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, readClkHold)
  ENUM_FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, dataMode, dataModesEnum)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, prstSel)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, acqDelay)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, intTime)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, digDelay)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, ampIdle)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, injTotal)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, rowColShiftPer)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, ampReset)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, digCount)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, digPeriod)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, biasTuning)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, pdpmndnmBalance)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, PeltierEnable)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, kpConstant)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, kiConstant)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, kdConstant)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, humidThold)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, setPoint)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ConfigV2QuadReg, ro)
  PyObject* digitalPots( PyObject* self, PyObject* );
  PyObject* gainMap( PyObject* self, PyObject* );

  PyMethodDef methods[] = {
    {"shiftSelect",     shiftSelect,    METH_NOARGS, "self.shiftSelect() -> int\n\nReturns integer number" },
    {"edgeSelect",      edgeSelect,     METH_NOARGS, "self.edgeSelect() -> int\n\nReturns integer number" },
    {"readClkSet",      readClkSet,     METH_NOARGS, "self.readClkSet() -> int\n\nReturns integer number" },
    {"readClkHold",     readClkHold,    METH_NOARGS, "self.readClkHold() -> int\n\nReturns integer number" },
    {"dataMode",        dataMode,       METH_NOARGS, "self.dataMode() -> DataModes enum\n\nReturns :py:class:`ConfigV2QuadReg.DataModes` enum" },
    {"prstSel",         prstSel,        METH_NOARGS, "self.prstSel() -> int\n\nReturns integer number" },
    {"acqDelay",        acqDelay,       METH_NOARGS, "self.acqDelay() -> int\n\nReturns integer number" },
    {"intTime",         intTime,        METH_NOARGS, "self.intTime() -> int\n\nReturns integer number" },
    {"digDelay",        digDelay,       METH_NOARGS, "self.digDelay() -> int\n\nReturns integer number" },
    {"ampIdle",         ampIdle,        METH_NOARGS, "self.ampIdle() -> int\n\nReturns integer number" },
    {"injTotal",        injTotal,       METH_NOARGS, "self.injTotal() -> int\n\nReturns integer number" },
    {"rowColShiftPer",  rowColShiftPer, METH_NOARGS, "self.rowColShiftPer() -> int\n\nReturns integer number" },
    {"ampReset",        ampReset,       METH_NOARGS, "self.ampReset() -> int\n\nReturns integer number" },
    {"digCount",        digCount,       METH_NOARGS, "self.digCount() -> int\n\nReturns integer number" },
    {"digPeriod",       digPeriod,      METH_NOARGS, "self.digPeriod() -> int\n\nReturns integer number" },
    {"biasTuning",      biasTuning,     METH_NOARGS, "self.biasTuning() -> int\n\nReturns integer number" },
    {"pdpmndnmBalance", pdpmndnmBalance,METH_NOARGS, "self.pdpmndnmBalance() -> int\n\nReturns integer number" },
    {"PeltierEnable",   PeltierEnable,  METH_NOARGS, "self.PeltierEnable() -> int\n\nReturns integer number" },
    {"kpConstant",      kpConstant,     METH_NOARGS, "self.kpConstant() -> int\n\nReturns integer number" },
    {"kiConstant",      kiConstant,     METH_NOARGS, "self.kiConstant() -> int\n\nReturns integer number" },
    {"kdConstant",      kdConstant,     METH_NOARGS, "self.kdConstant() -> int\n\nReturns integer number" },
    {"humidThold",      humidThold,     METH_NOARGS, "self.humidThold() -> int\n\nReturns integer number" },
    {"setPoint",        setPoint,       METH_NOARGS, "self.setPoint() -> int\n\nReturns integer number" },
    {"ro",              ro,             METH_NOARGS, "self.ro() -> CsPadReadOnlyCfg\n\nReturns :py:class:`CsPadReadOnlyCfg` object" },
    {"dp",              digitalPots,    METH_NOARGS, "self.dp() -> CsPadDigitalPotsCfg\n\nReturns :py:class:`CsPadDigitalPotsCfg` object" },
    {"gm",              gainMap,        METH_NOARGS, "self.gm() -> CsPadGainMapCfg\n\nReturns :py:class:`CsPadGainMapCfg` object" },
    {"readOnly",        ro,             METH_NOARGS, "self.readOnly() -> CsPadReadOnlyCfg\n\nReturns :py:class:`CsPadReadOnlyCfg` object, same as ro()" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad2x2::ConfigV2QuadReg class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad2x2::ConfigV2QuadReg::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "DataModes", dataModesEnum.type() );
  PyObject* val = PyInt_FromLong(Pds::CsPad2x2::TwoByTwosPerQuad);
  PyDict_SetItemString( type->tp_dict, "TwoByTwosPerQuad", val );
  Py_XDECREF(val);

  BaseType::initType( "ConfigV2QuadReg", module );
}

void
pypdsdata::CsPad2x2::ConfigV2QuadReg::print(std::ostream& str) const
{
  str << "cspad2x2.ConfigV2QuadReg(shiftSelect=" << m_obj->shiftSelect()
      << ", edgeSelect=" << m_obj->edgeSelect()
      << ", readClkSet=" << m_obj->readClkSet()
      << ", readClkHold=" << m_obj->readClkHold()
      << ", dataMode=" << m_obj->dataMode()
      << ", ...)";
}

namespace {

PyObject*
digitalPots( PyObject* self, PyObject* )
{
  Pds::CsPad2x2::ConfigV2QuadReg* obj = pypdsdata::CsPad2x2::ConfigV2QuadReg::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::CsPad2x2::CsPad2x2DigitalPotsCfg::PyObject_FromPds(
      const_cast<Pds::CsPad2x2::CsPad2x2DigitalPotsCfg*>(&obj->dp()), self, sizeof(Pds::CsPad2x2::CsPad2x2DigitalPotsCfg) );
}

PyObject*
gainMap( PyObject* self, PyObject* )
{
  Pds::CsPad2x2::ConfigV2QuadReg* obj = pypdsdata::CsPad2x2::ConfigV2QuadReg::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::CsPad2x2::CsPad2x2GainMapCfg::PyObject_FromPds(
      const_cast<Pds::CsPad2x2::CsPad2x2GainMapCfg*>(&obj->gm()), self, sizeof(Pds::CsPad2x2::CsPad2x2GainMapCfg) );
}

}
