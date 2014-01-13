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
  
  AliasMap amap;
  Src src_no_src;
  DetInfo dinfo1(0, DetInfo::Detector(0), 0, DetInfo::Device(0), 0);
  DetInfo dinfo2(1, DetInfo::Detector(1), 1, DetInfo::Device(1), 1);
  ProcInfo pinfo1(Pds::Level::Control, 0, 0x01010101);
  ProcInfo pinfo2(Pds::Level::Control, 0, 0x70707070);
  BldInfo binfo1(0, BldInfo::EBeam);
  BldInfo binfo2(0, BldInfo::PhaseCavity);

  
  void checkAny(Source xsrc)
  {
    Source::SrcMatch src = xsrc.srcMatch(amap);

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
  
  void checkNoSrc(Source xsrc)
  {
    Source::SrcMatch src = xsrc.srcMatch(amap);

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
  
  void checkAnyDetInfo(Source xsrc)
  {
    Source::SrcMatch src = xsrc.srcMatch(amap);

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

  void checkDetInfo0000(Source xsrc)
  {
    Source::SrcMatch src = xsrc.srcMatch(amap);

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
  
  void checkDetInfo1111(Source xsrc)
  {
    Source::SrcMatch src = xsrc.srcMatch(amap);

    BOOST_CHECK( not src.isNoSource() ) ;
    BOOST_CHECK( src.isExact() ) ;

    BOOST_CHECK( not src.match(src_no_src) ) ;

    BOOST_CHECK( not src.match(dinfo1) ) ;
    BOOST_CHECK( src.match(dinfo2) ) ;

    BOOST_CHECK( not src.match(pinfo1) ) ;
    BOOST_CHECK( not src.match(pinfo2) ) ;

    BOOST_CHECK( not src.match(binfo1) ) ;
    BOOST_CHECK( not src.match(binfo2) ) ;
  }

  void checkDetInfo(Source xsrc, Pds::DetInfo::Detector det, unsigned detId, Pds::DetInfo::Device dev, unsigned devId)
  {
    Source::SrcMatch src = xsrc.srcMatch(amap);

    BOOST_CHECK( not src.isNoSource() ) ;
    BOOST_CHECK( src.isExact() ) ;

    const Pds::Src& psrc = src.src();
    BOOST_CHECK_EQUAL(psrc.level(), Pds::Level::Source) ;

    const Pds::DetInfo& deti = static_cast<const Pds::DetInfo&>(psrc);
    BOOST_CHECK_EQUAL(deti.detector(), det) ;
    BOOST_CHECK_EQUAL(deti.detId(), detId) ;
    BOOST_CHECK_EQUAL(deti.device(), dev) ;
    BOOST_CHECK_EQUAL(deti.devId(), devId) ;
  }

  void checkDetInfoMatch(Source xsrc)
  {
    Source::SrcMatch src = xsrc.srcMatch(amap);

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
  
  void checkBldInfoEBeam(Source xsrc)
  {
    Source::SrcMatch src = xsrc.srcMatch(amap);

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
  
  void checkBldInfo(Source xsrc, Pds::BldInfo::Type type)
  {
    Source::SrcMatch src = xsrc.srcMatch(amap);

    BOOST_CHECK( not src.isNoSource() ) ;
    BOOST_CHECK( src.isExact() ) ;

    const Pds::Src& psrc = src.src();
    BOOST_CHECK_EQUAL(psrc.level(), Pds::Level::Reporter) ;

    const Pds::BldInfo& bld = static_cast<const Pds::BldInfo&>(psrc);
    BOOST_CHECK_EQUAL(bld.type(), type) ;
  }

  void checkAnyBldInfo(Source xsrc)
  {
    Source::SrcMatch src = xsrc.srcMatch(amap);

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
  
  void checkAnyProcInfo(Source xsrc)
  {
    Source::SrcMatch src = xsrc.srcMatch(amap);

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
  
  void checkProcInfo1(Source xsrc)
  {
    Source::SrcMatch src = xsrc.srcMatch(amap);

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

  ::checkDetInfo(Source("DetInfo(NoDetector.0:NoDevice.0)"), Pds::DetInfo::NoDetector, 0, Pds::DetInfo::NoDevice, 0);
  ::checkDetInfo(Source("DetInfo(AmoGasdet.5:Princeton.9)"), Pds::DetInfo::AmoGasdet, 5, Pds::DetInfo::Princeton, 9);
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

  ::checkBldInfo(Source("BldInfo(EBeam)"), Pds::BldInfo::EBeam);
  ::checkBldInfo(Source("BldInfo(PhaseCavity)"), Pds::BldInfo::PhaseCavity);
  ::checkBldInfo(Source("BldInfo(FEEGasDetEnergy)"), Pds::BldInfo::FEEGasDetEnergy);
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
  BOOST_CHECK_NO_THROW(Source("DetInfo()").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(.:)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:.)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(.:.)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(|)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(-|)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(|-)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(-|-)").srcMatch(amap));
  
  BOOST_CHECK_NO_THROW(Source("DetInfo(NoDetector)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoIMS)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoGD)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoETOF)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoITOF)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoMBES)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoVMI)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoBPS)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(Camp)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(EpicsArch)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(BldEb)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(SxrBeamline)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(SxrEndstation)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppSb1Ipm)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppSb1Pim)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppMonPim)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppSb2Ipm)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppSb3Ipm)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppSb3Pim)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppSb4Pim)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppGon)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppLas)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XppEndstation)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(AmoEndstation)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiEndstation)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(XcsEndstation)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(MecEndstation)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiDg1)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiDg2)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiDg3)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiDg4)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiKb1)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiDs1)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiDs2)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiDsu)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(CxiSc1)").srcMatch(amap));
  
  BOOST_CHECK_NO_THROW(Source("DetInfo(:NoDevice)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Evr)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Acqiris)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Opal1000)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Tm6740)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:pnCCD)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Princeton)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Fccd)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Ipimb)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Encoder)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:Cspad)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:AcqTDC)").srcMatch(amap));

  BOOST_CHECK_NO_THROW(Source("DetInfo(.1:.2)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(.1:)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(:.2)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(*.1:*.2)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(*.1:*)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("DetInfo(*:*.2)").srcMatch(amap));

  BOOST_CHECK_NO_THROW(Source("DetInfo(*-1|*-2)").srcMatch(amap));
  
  BOOST_CHECK_NO_THROW(Source("BldInfo()").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("BldInfo(EBeam)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("BldInfo(PhaseCavity)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("BldInfo(FEEGasDetEnergy)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("BldInfo(NH2-SB1-IPM-01)").srcMatch(amap));
  
  BOOST_CHECK_NO_THROW(Source("ProcInfo()").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("ProcInfo(0.0.0.0)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("ProcInfo(1.1.1.1)").srcMatch(amap));
  BOOST_CHECK_NO_THROW(Source("ProcInfo(255.255.255.255)").srcMatch(amap));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_str_invalid )
{
  BOOST_CHECK_THROW(Source("G").srcMatch(amap), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("!").srcMatch(amap), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("Info()").srcMatch(amap), ExceptionSourceFormat);
  
  BOOST_CHECK_THROW(Source("DetInfo(Unknown)").srcMatch(amap), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("DetInfo(:Unknown)").srcMatch(amap), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("DetInfo(.Unknown)").srcMatch(amap), ExceptionSourceFormat);

  BOOST_CHECK_THROW(Source("BldInfo(Unknown)").srcMatch(amap), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("BldInfo(.EBeam)").srcMatch(amap), ExceptionSourceFormat);

  BOOST_CHECK_THROW(Source("ProcInfo(1)").srcMatch(amap), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("ProcInfo(1.1.1)").srcMatch(amap), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("ProcInfo(1.1.1.1.1)").srcMatch(amap), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("ProcInfo(1024.1.1.1)").srcMatch(amap), ExceptionSourceFormat);
  BOOST_CHECK_THROW(Source("ProcInfo(psimport.slac.stanford.edu)").srcMatch(amap), ExceptionSourceFormat);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_src_format )
{
  Source src;
  src = Source(Pds::DetInfo::NoDetector, 255, Pds::DetInfo::NoDevice, 0);
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(NoDetector.*:NoDevice.0)");
  src = Source(Pds::DetInfo::Detector(255), 255, Pds::DetInfo::NoDevice, 255);
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(*.*:NoDevice.*)");
  src = Source(Pds::DetInfo::NoDetector, 255, Pds::DetInfo::Device(255), 255);
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(NoDetector.*:*.*)");
  src = Source(Pds::DetInfo::NoDetector, 0, Pds::DetInfo::Device(255), 0);
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(NoDetector.0:*.0)");
  src = Source(Pds::DetInfo::NoDetector, 0, Pds::DetInfo::Device(255), 255);
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(NoDetector.0:*.*)");
  src = Source(Pds::DetInfo::Detector(255), 0, Pds::DetInfo::Device(255), 255);
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(*.0:*.*)");
  src = Source(Pds::DetInfo::NoDetector, 255, Pds::DetInfo::Device(255), 255);
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(NoDetector.*:*.*)");
  src = Source(Pds::DetInfo::Detector(255), 55, Pds::DetInfo::NoDevice, 0);
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(*.55:NoDevice.0)");
  src = Source(Pds::DetInfo::Detector(255), 1, Pds::DetInfo::NoDevice, 2);
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(*.1:NoDevice.2)");
  src = Source(Pds::DetInfo::Detector(255), 255, Pds::DetInfo::Device(255), 254);
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(*.*:*.254)");
  src = Source(Pds::DetInfo::Detector(255), 255, Pds::DetInfo::Device(255), 255);
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "DetInfo(*.*:*.*)");

  src = Source(Pds::BldInfo::EBeam);
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "BldInfo(EBeam)");
  src = Source(Pds::BldInfo::Type(0xffffffff));
  BOOST_CHECK_EQUAL(lexical_cast<std::string>(src), "BldInfo()");
}

// ==============================================================


BOOST_AUTO_TEST_CASE( test_src_alias )
{
  amap.add("dinfo1", dinfo1);
  amap.add("dinfo2", dinfo2);
  amap.add("pinfo1", pinfo1);
  amap.add("pinfo2", pinfo2);
  amap.add("binfo1", binfo1);
  amap.add("binfo2", binfo2);

  ::checkDetInfo0000(Source("dinfo1"));
  ::checkDetInfo1111(Source("dinfo2"));
  ::checkBldInfo(Source("binfo1"), Pds::BldInfo::EBeam);
  ::checkBldInfo(Source("binfo2"), Pds::BldInfo::PhaseCavity);
  ::checkProcInfo1(Source("pinfo1"));
  BOOST_CHECK_THROW(Source("pinfo3").srcMatch(amap), ExceptionSourceFormat);
}

// ==============================================================
