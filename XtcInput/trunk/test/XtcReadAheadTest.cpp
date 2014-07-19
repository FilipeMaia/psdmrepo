//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcReadAheadTest...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <boost/make_shared.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "AppUtils/AppBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdArg.h"
#include "AppUtils/AppCmdOpt.h"
#include "MsgLogger/MsgLogger.h"
#include "XtcInput/ChunkFileIterList.h"
#include "XtcInput/Dgram.h"
#include "XtcInput/Exceptions.h"
#include "XtcInput/XtcStreamDgIter.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  struct DgData{
    uint32_t timeSec;  // time in seconds
    Pds::TransitionId::Value tran;  // transition type
    uint32_t damage;   // damage mask
  };

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//
//  Application class
//
class XtcReadAheadTest : public AppUtils::AppBase {
public:

  // Constructor
  explicit XtcReadAheadTest ( const std::string& appName ) ;

  // destructor
  ~XtcReadAheadTest () {}

protected :

  /**
   *  Main method which runs the whole application
   */
  virtual int runApp () ;

  /// write a bunch of datagrams to a file
  void write(const std::string& fileName, const ::DgData* dgData);

  void test(const ::DgData* in, const ::DgData* out);
  void test1();
  void test2();
  void test3();
  void test4();
  void test5();
  void test6();

private:

  static Dgram::ptr makeDgram(const ::DgData& dgData);
  static int open(const std::string& fileName);
  static bool checkDg(Dgram::ptr dg, bool empty, const ::DgData& dgData);

  // more command line options and arguments
  AppUtils::AppCmdArg<std::string> m_pathArg ;

};

//----------------
// Constructors --
//----------------
XtcReadAheadTest::XtcReadAheadTest ( const std::string& appName )
  : AppUtils::AppBase( appName )
  , m_pathArg(parser(), "path", "test file name" )
{
}

/**
 *  Main method which runs the whole application
 */
int
XtcReadAheadTest::runApp ()
{
  test1();
  test2();
  test3();
  test4();
  test5();
  test6();

  // return 0 on success, other values for error (like main())
  return 0 ;
}

void
XtcReadAheadTest::test1()
{
  // Simple test for already-ordered data, they should come out
  // in exactly the same order
  MsgLog("test1", info, "running test1");

  ::DgData dgDataIn[] = {
      {1, Pds::TransitionId::L1Accept, 0},
      {2, Pds::TransitionId::L1Accept, 0},
      {3, Pds::TransitionId::L1Accept, 0},
      {4, Pds::TransitionId::L1Accept, 0},
      // EOD
      {0, Pds::TransitionId::Unknown, 0},
  };

  ::DgData dgDataOut[] = {
      dgDataIn[0],
      dgDataIn[1],
      dgDataIn[2],
      dgDataIn[3],
      // EOD
      {0, Pds::TransitionId::Unknown},
  };

  test(dgDataIn, dgDataOut);
}

void
XtcReadAheadTest::test2()
{
  // Simple test for reverse-ordered data
  MsgLog("test2", info, "running test2");

  ::DgData dgDataIn[] = {
      {4, Pds::TransitionId::L1Accept, 0},
      {3, Pds::TransitionId::L1Accept, 0},
      {2, Pds::TransitionId::L1Accept, 0},
      {1, Pds::TransitionId::L1Accept, 0},
      // EOD
      {0, Pds::TransitionId::Unknown},
  };

  ::DgData dgDataOut[] = {
      dgDataIn[3],
      dgDataIn[2],
      dgDataIn[1],
      dgDataIn[0],
      // EOD
      {0, Pds::TransitionId::Unknown},
  };

  test(dgDataIn, dgDataOut);
}

void
XtcReadAheadTest::test3()
{
  // Test for configure "drift"
  MsgLog("test3", info, "running test3");

  ::DgData dgDataIn[] = {
      {1, Pds::TransitionId::Configure, 0},
      {0, Pds::TransitionId::L1Accept, 0},
      {2, Pds::TransitionId::L1Accept, 0},
      {3, Pds::TransitionId::L1Accept, 0},
      {5, Pds::TransitionId::Unconfigure, 0},
      // EOD
      {0, Pds::TransitionId::Unknown, 0},
  };

  ::DgData dgDataOut[] = {
      dgDataIn[0],
      dgDataIn[1],
      dgDataIn[2],
      dgDataIn[3],
      dgDataIn[4],
      // EOD
      {0, Pds::TransitionId::Unknown},
  };

  test(dgDataIn, dgDataOut);
}

void
XtcReadAheadTest::test4()
{
  // Test for configure "drift", check that L1Accept cannot
  // move across other transition types
  MsgLog("test4", info, "running test4");

  ::DgData dgDataIn[] = {
      {0, Pds::TransitionId::Configure, 0},
      {3, Pds::TransitionId::L1Accept, 0},
      {6, Pds::TransitionId::L1Accept, 0},
      {7, Pds::TransitionId::L1Accept, 0},
      {4, Pds::TransitionId::Unconfigure, 0},
      {5, Pds::TransitionId::Configure, 0},
      {2, Pds::TransitionId::L1Accept, 0},
      {8, Pds::TransitionId::L1Accept, 0},
      {9, Pds::TransitionId::L1Accept, 0},
      {7, Pds::TransitionId::Unconfigure, 0},
      // EOD
      {0, Pds::TransitionId::Unknown, 0},
  };

  ::DgData dgDataOut[] = {
      dgDataIn[0],
      dgDataIn[1],
      dgDataIn[2],
      dgDataIn[3],
      dgDataIn[4],
      dgDataIn[5],
      dgDataIn[6],
      dgDataIn[7],
      dgDataIn[8],
      dgDataIn[9],
      // EOD
      {0, Pds::TransitionId::Unknown},
  };

  test(dgDataIn, dgDataOut);
}

void
XtcReadAheadTest::test5()
{
  // Test for split transitions
  MsgLog("test5", info, "running test5");

  ::DgData dgDataIn[] = {
      {1, Pds::TransitionId::L1Accept, 0},
      {2, Pds::TransitionId::L1Accept, 1 << Pds::Damage::DroppedContribution},
      {3, Pds::TransitionId::L1Accept, 0},
      {2, Pds::TransitionId::L1Accept, 1 << Pds::Damage::DroppedContribution},
      // EOD
      {0, Pds::TransitionId::Unknown, 0},
  };

  ::DgData dgDataOut[] = {
      dgDataIn[0],
      dgDataIn[1],
      dgDataIn[3],
      dgDataIn[2],
      // EOD
      {0, Pds::TransitionId::Unknown},
  };

  test(dgDataIn, dgDataOut);
}

void
XtcReadAheadTest::test6()
{
  // Test for split transitions, check that split transitions
  // can move from one group of L1Accpets to another
  MsgLog("test6", info, "running test6");

  ::DgData dgDataIn[] = {
      {0, Pds::TransitionId::Configure, 0},
      {3, Pds::TransitionId::L1Accept, 1 << Pds::Damage::DroppedContribution},
      {6, Pds::TransitionId::L1Accept, 0},
      {7, Pds::TransitionId::L1Accept, 0},
      {4, Pds::TransitionId::Unconfigure, 0},
      {5, Pds::TransitionId::Configure, 0},
      {2, Pds::TransitionId::L1Accept, 1 << Pds::Damage::DroppedContribution},
      {8, Pds::TransitionId::L1Accept, 0},
      {9, Pds::TransitionId::L1Accept, 0},
      {7, Pds::TransitionId::Unconfigure, 0},
      // EOD
      {0, Pds::TransitionId::Unknown, 0},
  };

  ::DgData dgDataOut[] = {
      dgDataIn[0],
      dgDataIn[1],
      dgDataIn[6],
      dgDataIn[2],
      dgDataIn[3],
      dgDataIn[4],
      dgDataIn[5],
      dgDataIn[7],
      dgDataIn[8],
      dgDataIn[9],
      // EOD
      {0, Pds::TransitionId::Unknown},
  };

  test(dgDataIn, dgDataOut);
}

void
XtcReadAheadTest::test(const ::DgData* in, const ::DgData* out)
{
  std::string fname = m_pathArg.value();
  write(fname, in);

  XtcFileName files[1] = { XtcFileName(fname) };
  boost::shared_ptr<ChunkFileIterI> chunkFIter = boost::make_shared<ChunkFileIterList>(files+0, files+1);
  bool clockSort = true;
  XtcStreamDgIter iter(chunkFIter, clockSort);
  for ( ; out->timeSec; ++ out) {
    Dgram dg = iter.next();
    if (not checkDg(dg.dg(), false, *out)) return;
  }
  Dgram dg = iter.next();
  if (not checkDg(dg.dg(), true, *out)) return;

  unlink(fname.c_str());
}

bool
XtcReadAheadTest::checkDg(Dgram::ptr dg, bool empty, const ::DgData& dgData)
{
  if (not empty and not dg) {
    MsgLog("test1", error, "expected non-empty datagram, got empty");
    return false;
  }
  if (empty and dg) {
    MsgLog("test1", error, "expected empty datagram, got non-empty");
    return false;
  }
  if (not empty) {
    if (dg->seq.clock().seconds() != dgData.timeSec) {
      MsgLog("test1", error, "expected time " << dgData.timeSec << ", got " << dg->seq.clock().seconds());
      return false;
   }
    if (dg->seq.service() != dgData.tran) {
      MsgLog("test1", error, "expected transition " << dgData.tran << ", got " << dg->seq.service());
      return false;
   }
  }
  return true;
}


// function that will write a number of datagrams to output file
void
XtcReadAheadTest::write(const std::string& fileName, const ::DgData* dgData)
{
  int fd = open(fileName);
  if (fd < 0) return;

  for ( ; dgData->timeSec; ++ dgData) {

    Dgram::ptr dg = makeDgram(*dgData);
    ::write(fd, (char*)dg.get(), sizeof(Pds::Dgram)+dg->xtc.sizeofPayload());

  }
  close(fd);
}

int
XtcReadAheadTest::open(const std::string& fileName)
{
  int fd = ::open(fileName.c_str(), O_CREAT|O_TRUNC|O_WRONLY|O_SYNC, 0660);
  if (fd < 0) {
    MsgLog("writer", error, "Failed to open output file: " << fileName);
  }
  return fd;
}

Dgram::ptr
XtcReadAheadTest::makeDgram(const ::DgData& dgData)
{
  const size_t payloadSize = 10;
  char* buf = new char[sizeof(Pds::Dgram) + payloadSize];
  Pds::Dgram* dg = (Pds::Dgram*)buf;
  std::fill_n(buf, sizeof(Pds::Dgram) + payloadSize, '\xff');

  dg->seq = Pds::Sequence(Pds::Sequence::Event, dgData.tran, Pds::ClockTime(dgData.timeSec, 0), Pds::TimeStamp());
  dg->env = Pds::Env(1100);
  dg->xtc.damage = Pds::Damage(dgData.damage);
  dg->xtc.src = Pds::Src();
  dg->xtc.contains = Pds::TypeId(Pds::TypeId::Any, 0);
  dg->xtc.extent = payloadSize+sizeof(Pds::Xtc);

  return Dgram::make_ptr(dg);
}

} // namespace XtcInput


// this defines main()
APPUTILS_MAIN(XtcInput::XtcReadAheadTest)
