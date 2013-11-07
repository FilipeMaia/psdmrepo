//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcChunkDgIterTest...
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
#include <boost/thread.hpp>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

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
#include "XtcInput/XtcChunkDgIter.h"
#include "XtcInput/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//
//  Application class
//
class XtcChunkDgIterTest : public AppUtils::AppBase {
public:

  // Constructor
  explicit XtcChunkDgIterTest ( const std::string& appName ) ;

  // destructor
  ~XtcChunkDgIterTest () ;

protected :

  /**
   *  Main method which runs the whole application
   */
  virtual int runApp () ;

  static void writer1(int ndg, std::string fileName);
  static void writer2(int ndg, std::string fileName, std::string finalName, int timeout);
  static void writer3(int ndg, std::string fileName, std::string finalName, int timeout);

  void test1();
  void test2();
  void test3();
  void test4();
  void test5();

private:

  static Dgram::ptr makeDgram(size_t payloadSize);
  static int open(std::string fileName);
  static bool checkDg(const boost::shared_ptr<DgHeader>& hptr, bool empty, int payload);

  // more command line options and arguments
  AppUtils::AppCmdOpt<int> m_timeoutOpt ;
  AppUtils::AppCmdArg<std::string> m_pathArg ;

};

//----------------
// Constructors --
//----------------
XtcChunkDgIterTest::XtcChunkDgIterTest ( const std::string& appName )
  : AppUtils::AppBase( appName )
  , m_timeoutOpt(parser(), "t,timeout", "number", "timeout in seconds, def: 10", 10)
  , m_pathArg(parser(), "path", "test file name" )
{
}

//--------------
// Destructor --
//--------------
XtcChunkDgIterTest::~XtcChunkDgIterTest ()
{
}

/**
 *  Main method which runs the whole application
 */
int
XtcChunkDgIterTest::runApp ()
{
  test1();
  test2();
  test3();
  test4();
  test5();


  // return 0 on success, other values for error (like main())
  return 0 ;
}

void
XtcChunkDgIterTest::test1()
{
  // Simple test for non-live data reading,
  // write a bunch of datagrams and read them back,
  // check their sizes

  MsgLog("test1", info, "running test1");

  std::string fname = m_pathArg.value();
  writer1(5, fname);

  XtcChunkDgIter iter(XtcFileName(fname), 0);
  boost::shared_ptr<DgHeader> hptr;
  hptr = iter.next();
  if (not checkDg(hptr, false, 100)) return;
  hptr = iter.next();
  if (not checkDg(hptr, false, 110)) return;
  hptr = iter.next();
  if (not checkDg(hptr, false, 120)) return;
  hptr = iter.next();
  if (not checkDg(hptr, false, 130)) return;
  hptr = iter.next();
  if (not checkDg(hptr, false, 140)) return;
  hptr = iter.next();
  if (not checkDg(hptr, true, 0)) return;

  unlink(fname.c_str());
}

void
XtcChunkDgIterTest::test2()
{
  // Test for hang during live data reading,
  // write ".inprogress" file but do not rename it
  // expect timeout exception from reader

  MsgLog("test2", info, "running test2");

  std::string fname = m_pathArg.value() + ".inprogress";
  writer1(5, fname);

  XtcChunkDgIter iter(XtcFileName(fname), 3);
  boost::shared_ptr<DgHeader> hptr;
  hptr = iter.next();
  if (not checkDg(hptr, false, 100)) return;
  hptr = iter.next();
  if (not checkDg(hptr, false, 110)) return;
  hptr = iter.next();
  if (not checkDg(hptr, false, 120)) return;
  hptr = iter.next();
  if (not checkDg(hptr, false, 130)) return;
  hptr = iter.next();
  if (not checkDg(hptr, false, 140)) return;
  try {
    hptr = iter.next();
    MsgLog("test2", error, "did not receive expected timeout exception");
  } catch (const XTCLiveTimeout& exc) {
  }

  unlink(fname.c_str());
}

void
XtcChunkDgIterTest::test3()
{
  // Test for regular live data reading,
  // write ".inprogress" file in a separate thread
  // rename it to a final ".xtc" name.
  // Writer2 writes complete datagrams.

  MsgLog("test3", info, "running test3");

  std::string fname = m_pathArg.value() + ".inprogress";
  std::string finalname = m_pathArg.value();

  boost::thread thread(writer2, 3, fname, finalname, 3);

  sleep(1);

  XtcChunkDgIter iter(XtcFileName(fname), 6);
  boost::shared_ptr<DgHeader> hptr;
  hptr = iter.next();
  if (not checkDg(hptr, false, 100)) return;
  hptr = iter.next();
  if (not checkDg(hptr, false, 110)) return;
  hptr = iter.next();
  if (not checkDg(hptr, false, 120)) return;
  hptr = iter.next();
  if (not checkDg(hptr, true, 0)) return;

  thread.join();
  unlink(finalname.c_str());
}

void
XtcChunkDgIterTest::test4()
{
  // Test for regular live data reading,
  // write ".inprogress" file in a separate thread
  // rename it to a final ".xtc" name
  // Writer2 implements "slow" writing so that we can
  // read partial datagrams and wait for complete datagram

  MsgLog("test4", info, "running test4");

  std::string fname = m_pathArg.value() + ".inprogress";
  std::string finalname = m_pathArg.value();

  boost::thread thread(writer3, 3, fname, finalname, 1);

  sleep(1);

  XtcChunkDgIter iter(XtcFileName(fname), 5);
  boost::shared_ptr<DgHeader> hptr;
  hptr = iter.next();
  if (not checkDg(hptr, false, 100)) return;
  hptr = iter.next();
  if (not checkDg(hptr, false, 110)) return;
  hptr = iter.next();
  if (not checkDg(hptr, false, 120)) return;
  hptr = iter.next();
  if (not checkDg(hptr, true, 0)) return;

  thread.join();
  unlink(finalname.c_str());
}

void
XtcChunkDgIterTest::test5()
{
  // Test for failed transfer,
  // write ".inprogress" file in a separate thread
  // rename it to a final ".inprogress.XXXX"

  MsgLog("test5", info, "running test5");

  std::string fname = m_pathArg.value() + ".inprogress";
  std::string finalname = m_pathArg.value() + ".inprogress.123";

  boost::thread thread(writer3, 3, fname, finalname, 1);

  sleep(1);

  XtcChunkDgIter iter(XtcFileName(fname), 5);
  boost::shared_ptr<DgHeader> hptr;
  hptr = iter.next();
  if (not checkDg(hptr, false, 100)) return;
  hptr = iter.next();
  if (not checkDg(hptr, false, 110)) return;
  hptr = iter.next();
  if (not checkDg(hptr, false, 120)) return;
  try {
    hptr = iter.next();
    MsgLog("test5", error, "did not receive expected timeout exception");
  } catch (const XTCLiveTimeout& exc) {
  }

  thread.join();
  unlink(finalname.c_str());
}

bool
XtcChunkDgIterTest::checkDg(const boost::shared_ptr<DgHeader>& hptr, bool empty, int payload)
{
  if (not empty and not hptr) {
    MsgLog("test1", error, "expected non-empty datagram, got empty");
    return false;
  }
  if (empty and hptr) {
    MsgLog("test1", error, "expected empty datagram, got non-empty");
    return false;
  }
  if (not empty) {
    Dgram::ptr dg = hptr->dgram();
    if (dg->xtc.sizeofPayload() != payload) {
      MsgLog("test1", error, "expected payload size " << payload << ", got " << dg->xtc.sizeofPayload());
      return false;
   }
  }
  return true;
}


// function that will write a number of datagrams to output file
void
XtcChunkDgIterTest::writer1(int ndg, std::string fileName)
{
  int fd = open(fileName);
  if (fd < 0) return;

  for (int i = 0; i < ndg; ++ i) {

    size_t payloadSize = 10*i + 100;
    Dgram::ptr dg = makeDgram(payloadSize);
    write(fd, (char*)dg.get(), sizeof(Pds::Dgram)+dg->xtc.sizeofPayload());

  }
  close(fd);
}

// function that will write a number of datagrams to output file then renames it after timeout
void
XtcChunkDgIterTest::writer2(int ndg, std::string fileName, std::string finalName, int timeout)
{
  writer1(ndg, fileName);

  sleep(timeout);
  rename(fileName.c_str(), finalName.c_str());
}

// function that slowly writes a number of datagrams in "slow" mode
void
XtcChunkDgIterTest::writer3(int ndg, std::string fileName, std::string finalName, int timeout)
{
  int fd = open(fileName);
  if (fd < 0) return;

  for (int i = 0; i < ndg; ++ i) {

    size_t payloadSize = 10*i + 100;
    Dgram::ptr dg = makeDgram(payloadSize);

    char* b = (char*)dg.get();
    size_t size = sizeof(Pds::Dgram)+dg->xtc.sizeofPayload();

    size_t off = 0;
    while (off < size) {
      size_t s = size - off;
      if (off == 0) s = 15;
      if (off == 15) s = 40;
      off += write(fd, b + off, s);
      sleep(1);
    }

  }
  close(fd);

  sleep(timeout);
  rename(fileName.c_str(), finalName.c_str());
}

int
XtcChunkDgIterTest::open(std::string fileName)
{
  int fd = ::open(fileName.c_str(), O_CREAT|O_TRUNC|O_WRONLY|O_SYNC, 0660);
  if (fd < 0) {
    MsgLog("writer", error, "Failed to open output file: " << fileName);
  }
  return fd;
}

Dgram::ptr
XtcChunkDgIterTest::makeDgram(size_t payloadSize)
{
  char* buf = new char[sizeof(Pds::Dgram) + payloadSize];
  Pds::Dgram* dg = (Pds::Dgram*)buf;
  std::fill_n(buf, sizeof(Pds::Dgram) + payloadSize, '\xff');

  dg->seq = Pds::Sequence(Pds::ClockTime(1,1), Pds::TimeStamp());
  dg->env = Pds::Env(1100);
  dg->xtc.damage = Pds::Damage(13);
  dg->xtc.src = Pds::Src();
  dg->xtc.contains = Pds::TypeId(Pds::TypeId::Any, 0);
  dg->xtc.extent = payloadSize+sizeof(Pds::Xtc);

  return Dgram::make_ptr(dg);
}

} // namespace XtcInput


// this defines main()
APPUTILS_MAIN(XtcInput::XtcChunkDgIterTest)
