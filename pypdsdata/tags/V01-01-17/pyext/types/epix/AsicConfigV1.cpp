//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AsicConfigV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "AsicConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, monostPulser)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, dummyTest)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, dummyMask)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, pulser)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, pbit)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, atest)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, test)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, sabTest)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, hrTest)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, digMon1)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, digMon2)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, pulserDac)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, Dm1En)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, Dm2En)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, slvdSBit)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, VRefDac)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, TpsTComp)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, TpsMux)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, RoMonost)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, TpsGr)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, S2dGr)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, PpOcbS2d)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, Ocb)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, Monost)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, FastppEnable)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, Preamp)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, PixelCb)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, S2dTComp)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, FilterDac)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, TC)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, S2d)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, S2dDacBias)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, TpsTcDac)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, TpsDac)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, S2dTcDac)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, S2dDac)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, TestBe)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, IsEn)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, DelExec)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, DelCckReg)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, RowStartAddr)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, RowStopAddr)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, ColStartAddr)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, ColStopAddr)
  FUN0_WRAPPER(pypdsdata::Epix::AsicConfigV1, chipID)

  PyMethodDef methods[] = {
    { "monostPulser",    monostPulser,    METH_NOARGS, "self.monostPulser() -> int\n\nReturns integer number" },
    { "dummyTest",       dummyTest,       METH_NOARGS, "self.dummyTest() -> int\n\nReturns integer number" },
    { "dummyMask",       dummyMask,       METH_NOARGS, "self.dummyMask() -> int\n\nReturns integer number" },
    { "pulser",          pulser,          METH_NOARGS, "self.pulser() -> int\n\nReturns integer number" },
    { "pbit",            pbit,            METH_NOARGS, "self.pbit() -> int\n\nReturns integer number" },
    { "atest",           atest,           METH_NOARGS, "self.atest() -> int\n\nReturns integer number" },
    { "test",            test,            METH_NOARGS, "self.test() -> int\n\nReturns integer number" },
    { "sabTest",         sabTest,         METH_NOARGS, "self.sabTest() -> int\n\nReturns integer number" },
    { "hrTest",          hrTest,          METH_NOARGS, "self.hrTest() -> int\n\nReturns integer number" },
    { "digMon1",         digMon1,         METH_NOARGS, "self.digMon1() -> int\n\nReturns integer number" },
    { "digMon2",         digMon2,         METH_NOARGS, "self.digMon2() -> int\n\nReturns integer number" },
    { "pulserDac",       pulserDac,       METH_NOARGS, "self.pulserDac() -> int\n\nReturns integer number" },
    { "Dm1En",           Dm1En,           METH_NOARGS, "self.Dm1En() -> int\n\nReturns integer number" },
    { "Dm2En",           Dm2En,           METH_NOARGS, "self.Dm2En() -> int\n\nReturns integer number" },
    { "slvdSBit",        slvdSBit,        METH_NOARGS, "self.slvdSBit() -> int\n\nReturns integer number" },
    { "VRefDac",         VRefDac,         METH_NOARGS, "self.VRefDac() -> int\n\nReturns integer number" },
    { "TpsTComp",        TpsTComp,        METH_NOARGS, "self.TpsTComp() -> int\n\nReturns integer number" },
    { "TpsMux",          TpsMux,          METH_NOARGS, "self.TpsMux() -> int\n\nReturns integer number" },
    { "RoMonost",        RoMonost,        METH_NOARGS, "self.RoMonost() -> int\n\nReturns integer number" },
    { "TpsGr",           TpsGr,           METH_NOARGS, "self.TpsGr() -> int\n\nReturns integer number" },
    { "S2dGr",           S2dGr,           METH_NOARGS, "self.S2dGr() -> int\n\nReturns integer number" },
    { "PpOcbS2d",        PpOcbS2d,        METH_NOARGS, "self.PpOcbS2d() -> int\n\nReturns integer number" },
    { "Ocb",             Ocb,             METH_NOARGS, "self.Ocb() -> int\n\nReturns integer number" },
    { "Monost",          Monost,          METH_NOARGS, "self.Monost() -> int\n\nReturns integer number" },
    { "FastppEnable",    FastppEnable,    METH_NOARGS, "self.FastppEnable() -> int\n\nReturns integer number" },
    { "Preamp",          Preamp,          METH_NOARGS, "self.Preamp() -> int\n\nReturns integer number" },
    { "PixelCb",         PixelCb,         METH_NOARGS, "self.PixelCb() -> int\n\nReturns integer number" },
    { "S2dTComp",        S2dTComp,        METH_NOARGS, "self.S2dTComp() -> int\n\nReturns integer number" },
    { "FilterDac",       FilterDac,       METH_NOARGS, "self.FilterDac() -> int\n\nReturns integer number" },
    { "TC",              TC,              METH_NOARGS, "self.TC() -> int\n\nReturns integer number" },
    { "S2d",             S2d,             METH_NOARGS, "self.S2d() -> int\n\nReturns integer number" },
    { "S2dDacBias",      S2dDacBias,      METH_NOARGS, "self.S2dDacBias() -> int\n\nReturns integer number" },
    { "TpsTcDac",        TpsTcDac,        METH_NOARGS, "self.TpsTcDac() -> int\n\nReturns integer number" },
    { "TpsDac",          TpsDac,          METH_NOARGS, "self.TpsDac() -> int\n\nReturns integer number" },
    { "S2dTcDac",        S2dTcDac,        METH_NOARGS, "self.S2dTcDac() -> int\n\nReturns integer number" },
    { "S2dDac",          S2dDac,          METH_NOARGS, "self.S2dDac() -> int\n\nReturns integer number" },
    { "TestBe",          TestBe,          METH_NOARGS, "self.TestBe() -> int\n\nReturns integer number" },
    { "IsEn",            IsEn,            METH_NOARGS, "self.IsEn() -> int\n\nReturns integer number" },
    { "DelExec",         DelExec,         METH_NOARGS, "self.DelExec() -> int\n\nReturns integer number" },
    { "DelCckReg",       DelCckReg,       METH_NOARGS, "self.DelCckReg() -> int\n\nReturns integer number" },
    { "RowStartAddr",    RowStartAddr,    METH_NOARGS, "self.RowStartAddr() -> int\n\nReturns integer number" },
    { "RowStopAddr",     RowStopAddr,     METH_NOARGS, "self.RowStopAddr() -> int\n\nReturns integer number" },
    { "ColStartAddr",    ColStartAddr,    METH_NOARGS, "self.ColStartAddr() -> int\n\nReturns integer number" },
    { "ColStopAddr",     ColStopAddr,     METH_NOARGS, "self.ColStopAddr() -> int\n\nReturns integer number" },
    { "chipID",          chipID,          METH_NOARGS, "self.chipID() -> int\n\nReturns integer number" },
    {0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Epix::AsicConfigV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Epix::AsicConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "AsicConfigV1", module );
}

void
pypdsdata::Epix::AsicConfigV1::print(std::ostream& str) const
{
  str << "Epix.AsicConfigV1(monostPulser=" << int(m_obj->monostPulser())
      << ", dummyTest=" << int(m_obj->dummyTest())
      << ", dummyMask=" << int(m_obj->dummyMask())
      << ", pulser=" << int(m_obj->pulser())
      << ", pbit=" << int(m_obj->pbit())
      << ", atest=" << int(m_obj->atest())
      << ", test=" << int(m_obj->test())
      << ", chipID=" << int(m_obj->chipID())
      << ", ...)" ;
}
