//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the PSEvt::Source.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------
#include <boost/lexical_cast.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/Source.h"
#include "PSEvt/Exceptions.h"

using namespace PSEvt;
using namespace Pds;
using namespace boost;

#define BOOST_TEST_MODULE psevt-source-unit-test
#include <boost/test/included/unit_test.hpp>

namespace {
  
  Src src_no_src;
  DetInfo dinfo1(0, DetInfo::Detector(0), 0, DetInfo::Device(0), 0);
  DetInfo dinfo2(1, DetInfo::Detector(1), 1, DetInfo::Device(1), 1);
  ProcInfo pinfo1(Pds::Level::Control, 0, 0x01010101);
  ProcInfo pinfo2(Pds::Level::Control, 0, 0x70707070);
  BldInfo binfo1(0, BldInfo::EBeam);
  BldInfo binfo2(0, BldInfo::PhaseCavity);

  
  void checkAny(Source src)
  {
    BOOST_CHECK( not src.isNoSource() ) ;
    BOOST_CHECK( not src.isExact() ) ;

    BOOST_CHECK( src.match(src_no_src) ) ;

    BOOST_CHECK( src.match(dinfo1) ) ;
    BOOST_CHECK( src.match(dinfo2) ) ;

    BOOST_CHECK( src.match(pinfo1) ) ;
    BOOST_CHECK( src.match(pinfo2) ) ;

    BOOST_CHECK( src.match(binfo1) ) ;
    BOOST_CHECK( src.match(binfo2) ) ;
  }
  
  void checkNoSrc(Source src)
  {
    BOOST_CHECK( src.isNoSource() ) ;
    BOOST_CHECK( src.isExact() ) ;

    BOOST_CHECK( src.match(src_no_src) ) ;

    BOOST_CHECK( not src.match(dinfo1) ) ;
    BOOST_CHECK( not src.match(dinfo2) ) ;

    BOOST_CHECK( not src.match(pinfo1) ) ;
    BOOST_CHECK( not src.match(pinfo2) ) ;

    BOOST_CHECK( not src.match(binfo1) ) ;
    BOOST_CHECK( not src.match(binfo2) ) ;
  }
  
  void checkAnyDetInfo(Source src)
  {
    BOOST_CHECK( not src.isNoSource() ) ;
    BOOST_CHECK( not src.isExact() ) ;

    BOOST_CHECK( not src.match(src_no_src) ) ;

    BOOST_CHECK( src.match(dinfo1) ) ;
    BOOST_CHECK( src.match(dinfo2) ) ;

    BOOST_CHECK( not src.match(pinfo1) ) ;
    BOOST_CHECK( not src.match(pinfo2) ) ;

    BOOST_CHECK( not src.match(binfo1) ) ;
    BOOST_CHECK( not src.match(binfo2) ) ;
  }

  void checkDetInfo0000(Source src)
  {
    BOOST_CHECK( not src.isNoSource() ) ;
    BOOST_CHECK( src.isExact() ) ;

    BOOST_CHECK( not src.match(src_no_src) ) ;

    BOOST_CHECK( src.match(dinfo1) ) ;
    BOOST_CHECK( not src.match(dinfo2) ) ;

    BOOST_CHECK( not src.match(pinfo1) ) ;
    BOOST_CHECK( not src.match(pinfo2) ) ;

    BOOST_CHECK( not src.match(binfo1) ) ;
    BOOST_CHECK( not src.match(binfo2) ) ;
  }
  
  void checkDetInfoMatch(Source src)
  {
    BOOST_CHECK( not src.isNoSource() ) ;
    BOOST_CHECK( not src.isExact() ) ;

    BOOST_CHECK( not src.match(src_no_src) ) ;

    BOOST_CHECK( src.match(dinfo1) ) ;
    BOOST_CHECK( not src.match(dinfo2) ) ;

    BOOST_CHECK( not src.match(pinfo1) ) ;
    BOOST_CHECK( not src.match(pinfo2) ) ;

    BOOST_CHECK( not src.match(binfo1) ) ;
    BOOST_CHECK( not src.match(binfo2) ) ;
  }
  
  void checkBldInfoEBeam(Source src)
  {
    BOOST_CHECK( not src.isNoSource() ) ;
    BOOST_CHECK( src.isExact() ) ;

    BOOST_CHECK( not src.match(src_no_src) ) ;

    BOOST_CHECK( not src.match(dinfo1) ) ;
    BOOST_CHECK( not src.match(dinfo2) ) ;

    BOOST_CHECK( not src.match(pinfo1) ) ;
    BOOST_CHECK( not src.match(pinfo2) ) ;

    BOOST_CHECK( src.match(binfo1) ) ;
    BOOST_CHECK( not src.match(binfo2) ) ;
  }
  
  void checkAnyBldInfo(Source src)
  {
    BOOST_CHECK( not src.isNoSource() ) ;
    BOOST_CHECK( not src.isExact() ) ;

    BOOST_CHECK( not src.match(src_no_src) ) ;

    BOOST_CHECK( not src.match(dinfo1) ) ;
    BOOST_CHECK( not src.match(dinfo2) ) ;

    BOOST_CHECK( not src.match(pinfo1) ) ;
    BOOST_CHECK( not src.match(pinfo2) ) ;

    BOOST_CHECK( src.match(binfo1) ) ;
    BOOST_CHECK( src.match(binfo2) ) ;
  }
  
  void checkAnyProcInfo(Source src)
  {
    BOOST_CHECK( not src.isNoSource() ) ;
    BOOST_CHECK( not src.isExact() ) ;

    BOOST_CHECK( not src.match(src_no_src) ) ;

    BOOST_CHECK( not src.match(dinfo1) ) ;
    BOOST_CHECK( not src.match(dinfo2) ) ;

    BOOST_CHECK( src.match(pinfo1) ) ;
    BOOST_CHECK( src.match(pinfo2) ) ;

    BOOST_CHECK( not src.match(binfo1) ) ;
    BOOST_CHECK( not src.match(binfo2) ) ;
  }
  
  void checkProcInfo1(Source src)
  {
    BOOST_CHECK( not src.isNoSource() ) ;
    BOOST_CHECK( src.isExact() ) ;

    BOOST_CHECK( not src.match(src_no_src) ) ;

    BOOST_CHECK( not src.match(dinfo1) ) ;
    BOOST_CHECK( not src.match(dinfo2) ) ;

    BOOST_CHECK( src.match(pinfo1) ) ;
    BOOST_CHECK( not src.match(pinfo2) ) ;

    BOOST_CHECK( not src.match(binfo1) ) ;
    BOOST_CHECK( not src.match(binfo2) ) ;
  }
  
  
}


/**
 * Simple test suite for module psevt-source-unit-test.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

// ==============================================================

BOOST_AUTO_TEST_CASE( test_any )
{
  Source src;

  ::checkAny(src);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_no_source )
{
  Source src(Source::null);
  
  ::checkNoSrc(src);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_detinfo1 )
{
  Source src(DetInfo::Detector(0), 0, DetInfo::Device(0), 0);
  
  ::checkDetInfo0000(src);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_detinfo2 )
{
  Source src(DetInfo(0, DetInfo::Detector(0), 0, DetInfo::Device(0), 0));
  
  ::checkDetInfo0000(src);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_bldinfo1 )
{
  Source src(BldInfo::EBeam);
  
  ::checkBldInfoEBeam(src);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_bldinfo2 )
{
  Source src(BldInfo(0, BldInfo::EBeam));
  
  ::checkBldInfoEBeam(src);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_str_any )
{
  Source src("");

  ::checkAny(src);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_str_any_detinfo )
{
  ::checkAnyDetInfo(Source("DetInfo()"));
  ::checkAnyDetInfo(Source("DetInfo(*.*:*.*)"));
  ::checkAnyDetInfo(Source("DetInfo(*:*)"));
  ::checkAnyDetInfo(Source("DetInfo(:)"));
  ::checkAnyDetInfo(Source("DetInfo(.*:.*)"));
  ::checkAnyDetInfo(Source("DetInfo(*)"));
  ::checkAnyDetInfo(Source("DetInfo(:*)"));
  ::checkAnyDetInfo(Source("DetInfo(*-*|*-*)"));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_str_detinfo )
{
  ::checkDetInfo0000(Source("DetInfo(NoDetector.0:NoDevice.0)"));
  ::checkDetInfo0000(Source("NoDetector.0:NoDevice.0"));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_str_detinfo_det )
{
  ::checkDetInfoMatch(Source("DetInfo(NoDetector.*:NoDevice.0)"));
  ::checkDetInfoMatch(Source("DetInfo(:NoDevice)"));
  ::checkDetInfoMatch(Source("DetInfo(NoDetector)"));
  ::checkDetInfoMatch(Source("DetInfo(NoDetector.0:*.0)"));
  ::checkDetInfoMatch(Source("DetInfo(NoDetector.0:*.*)"));
  ::checkDetInfoMatch(Source("DetInfo(*.0:*.*)"));
  ::checkDetInfoMatch(Source("DetInfo(NoDetector.*:*.*)"));
  ::checkDetInfoMatch(Source("DetInfo(*.*:NoDevice.0)"));
  ::checkDetInfoMatch(Source("DetInfo(*.*:NoDevice.*)"));
  ::checkDetInfoMatch(Source("DetInfo(*.*:*.0)"));

  ::checkDetInfoMatch(Source("NoDetector.*:NoDevice.0"));
  ::checkDetInfoMatch(Source("NoDetector"));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_str_any_bldinfo )
{
  ::checkAnyBldInfo(Source("BldInfo()"));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_str_bldinfo )
{
  ::checkBldInfoEBeam(Source("BldInfo(EBeam)"));
  ::checkBldInfoEBeam(Source("EBeam"));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_str_any_procinfo )
{
  ::checkAnyProcInfo(Source("ProcInfo()"));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_str_procinfo )
{
  ::checkProcInfo1(Source("ProcInfo(1.1.1.1)"));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_str_valid )
{
  BOOST_CHECK_NO_THROW(Source("DetInfo()"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(.:)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:.)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(.:.)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(|)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(-|)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(|-)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(-|-)"));
  
  BOOST_CHECK_NO_THROW(Source("DetInfo(NoDetector)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoIMS)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoGD)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoETOF)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoITOF)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoMBES)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoVMI)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoBPS)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(Camp)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(EpicsArch)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(BldEb)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(SxrBeamline)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(SxrEndstation)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppSb1Ipm)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppSb1Pim)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppMonPim)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppSb2Ipm)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppSb3Ipm)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppSb3Pim)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppSb4Pim)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppGon)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppLas)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppEndstation)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoEndstation)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiEndstation)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XcsEndstation)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(MecEndstation)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiDg1)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiDg2)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiDg3)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiDg4)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiKb1)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiDs1)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiDs2)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiDsu)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiSc1)"));
  
  BOOST_CHECK_NO_THROW(Source("DetInfo(:NoDevice)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Evr)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Acqiris)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Opal1000)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Tm6740)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:pnCCD)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Princeton)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Fccd)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Ipimb)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Encoder)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Cspad)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:AcqTDC)"));

  BOOST_CHECK_NO_THROW(Source("DetInfo(.1:.2)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(.1:)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:.2)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(*.1:*.2)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(*.1:*)"));
  BOOST_CHECK_NO_THROW(Source("DetInfo(*:*.2)"));

  BOOST_CHECK_NO_THROW(Source("DetInfo(*-1|*-2)"));
  
  BOOST_CHECK_NO_THROW(Source("BldInfo()"));
  BOOST_CHECK_NO_THROW(Source("BldInfo(EBeam)"));
  BOOST_CHECK_NO_THROW(Source("BldInfo(PhaseCavity)"));
  BOOST_CHECK_NO_THROW(Source("BldInfo(FEEGasDetEnergy)"));
  BOOST_CHECK_NO_THROW(Source("BldInfo(NH2-SB1-IPM-01)"));
  
  BOOST_CHECK_NO_THROW(Source("ProcInfo()"));
  BOOST_CHECK_NO_THROW(Source("ProcInfo(0.0.0.0)"));
  BOOST_CHECK_NO_THROW(Source("ProcInfo(1.1.1.1)"));
  BOOST_CHECK_NO_THROW(Source("ProcInfo(255.255.255.255)"));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_str_invalid )
{
  BOOST_CHECK_THROW(Source("G"), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("!"), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("Info()"), ExceptionSourceFormat);
  
  BOOST_CHECK_THROW(Source("DetInfo(Unknown)"), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("DetInfo(:Unknown)"), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("DetInfo(.Unknown)"), ExceptionSourceFormat);

  BOOST_CHECK_THROW(Source("BldInfo(Unknown)"), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("BldInfo(.EBeam)"), ExceptionSourceFormat);

  BOOST_CHECK_THROW(Source("ProcInfo(1)"), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("ProcInfo(1.1.1)"), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("ProcInfo(1.1.1.1.1)"), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("ProcInfo(1024.1.1.1)"), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("ProcInfo(psimport.slac.stanford.edu)"), ExceptionSourceFormat);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_src_format )
{
  Source src;
  src = Source("DetInfo(NoDetector.*:NoDevice.0)");
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(NoDetector.*:NoDevice.0)");
  src = Source("DetInfo(:NoDevice)");
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(*.*:NoDevice.*)");
  src = Source("DetInfo(NoDetector)");
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(NoDetector.*:*.*)");
  src = Source("DetInfo(NoDetector.0:*.0)");
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(NoDetector.0:*.0)");
  src = Source("DetInfo(NoDetector.0:*.*)");
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(NoDetector.0:*.*)");
  src = Source("DetInfo(*.0:*.*)");
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(*.0:*.*)");
  src = Source("DetInfo(NoDetector.*:*.*)");
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(NoDetector.*:*.*)");
  src = Source("DetInfo(*.55:NoDevice.0)");
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(*.55:NoDevice.0)");
  src = Source("DetInfo(*.1:NoDevice.2)");
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(*.1:NoDevice.2)");
  src = Source("DetInfo(*.*:*.254)");
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(*.*:*.254)");
  src = Source("DetInfo(*.*:*.255)");
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(*.*:*.*)");

  src = Source("BldInfo(EBeam)");
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "BldInfo(EBeam)");
  src = Source("BldInfo()");
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "BldInfo()");
}

// ==============================================================

