/**
 * Original code for XTC scanning from Chris
 *
 * $Id$
 */
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <list>
#include <boost/make_shared.hpp>
#include <boost/thread/thread.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "AppUtils/AppBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdArgList.h"
#include "AppUtils/AppCmdOpt.h"
#include "AppUtils/AppCmdOptBool.h"
#include "AppUtils/AppCmdOptNamedValue.h"
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/xtc/BldInfo.hh"
#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/ProcInfo.hh"
#include "pdsdata/xtc/XtcIterator.hh"
#include "pdsdata/xtc/XtcFileIterator.hh"
#include "XtcInput/RunFileIterList.h"
#include "XtcInput/DgramQueue.h"
#include "XtcInput/DgramReader.h"
#include "XtcInput/XtcFileName.h"
#include "XtcInput/MergeMode.h"

//
//  XTC iterator class which dumps XTC-level info
//
class myLevelIter : public XtcIterator {
public:
  enum {Stop, Continue};
  myLevelIter(Xtc* xtc) : XtcIterator(xtc), _depth(1) {}
  int process(Xtc* xtc) {

    for ( unsigned i=0 ; i < _depth ; ++ i ) printf("  ");

    Level::Type level = xtc->src.level();

    printf("%s level: ", Level::name(level));
    printf("payloadSize=%d ", xtc->sizeofPayload() );
    printf("damage=%x ", xtc->damage.value() );

    if (level == Pds::Level::Source) {

      const DetInfo& info = static_cast<const DetInfo&>(xtc->src);
      printf("src=DetInfo(%s.%d:%s.%d)",
             DetInfo::name(info.detector()),info.detId(),
             DetInfo::name(info.device()),info.devId());

    } else if (level == Pds::Level::Reporter) {

      const BldInfo& info = static_cast<const BldInfo&>(xtc->src);
      printf("src=BldInfo(%s)", BldInfo::name(info));

    } else {

      const ProcInfo& info = static_cast<const ProcInfo&>(xtc->src);
      unsigned ip = info.ipAddr();
      printf("src=ProcInfo(%d.%d.%d.%d, pid=%d)", ((ip>>24)&0xff), ((ip>>16)&0xff),
          ((ip>>8)&0xff), (ip&0xff), info.processId());

    }

    if (xtc->contains.id() == TypeId::Id_Xtc ) {
      printf("\n");
      ++_depth;
      this->iterate( xtc );
      --_depth;
    } else {
      printf(" id=%d name=%s version=%d\n", xtc->contains.id(),
          Pds::TypeId::name(xtc->contains.id()), xtc->contains.version() );
    }
    return Continue;
  }
private:
  unsigned _depth;
};


namespace O2OTranslator {

using namespace AppUtils ;

//
//  Application class
//
class O2O_Scanner : public AppBase {
public:

  // Constructor
  explicit O2O_Scanner ( const std::string& appName ) ;

  // destructor
  ~O2O_Scanner () ;

protected :

  /**
   *  Main method which runs the whole application
   */
  virtual int runApp () ;

private:

  // more command line options and arguments
  AppCmdOptBool               m_skipDamaged ;
  AppCmdOpt<double>           m_l1offset ;
  AppCmdOptNamedValue<XtcInput::MergeMode> m_mergeMode ;
  AppCmdOpt<std::string>      m_liveDbConn;
  AppCmdOpt<std::string>      m_liveTable;
  AppCmdOpt<unsigned>         m_liveTimeout;
  AppCmdArgList<std::string>  m_inputFiles ;

};

//----------------
// Constructors --
//----------------
O2O_Scanner::O2O_Scanner ( const std::string& appName )
  : AppBase( appName )
  , m_skipDamaged( 'd', "skip-damaged",             "skip damaged datagrams", false )
  , m_l1offset   (      "l1-offset",    "number",   "L1Accept time offset seconds, def: 0", 0 )
  , m_mergeMode  ( 'j', "merge-mode",   "mode-name","one of one-stream, no-chunking, file-name; def: file-name", XtcInput::MergeFileName )
  , m_liveDbConn (      "live-db",      "string",   "database connection string for live database", "" )
  , m_liveTable  (      "live-table",   "string",   "table name for live database, def: file", "file" )
  , m_liveTimeout(      "live-timeout", "number",   "timeout for live data in seconds, def: 120", 120U )
  , m_inputFiles ( "input-xtc", "the list of the input XTC files" )
{
  addOption( m_skipDamaged ) ;
  addOption( m_l1offset ) ;
  addOption( m_mergeMode ) ;
  m_mergeMode.add ( "one-stream", XtcInput::MergeOneStream ) ;
  m_mergeMode.add ( "no-chunking", XtcInput::MergeNoChunking ) ;
  m_mergeMode.add ( "file-name", XtcInput::MergeFileName ) ;
  addOption( m_liveDbConn ) ;
  addOption( m_liveTable ) ;
  addOption( m_liveTimeout ) ;
  addArgument( m_inputFiles ) ;
}

//--------------
// Destructor --
//--------------
O2O_Scanner::~O2O_Scanner ()
{
}

/**
 *  Main method which runs the whole application
 */
int
O2O_Scanner::runApp ()
{
  if (m_inputFiles.empty()) {
    MsgLogRoot(error, "no input data files specified" ) ;
    return 2 ;
  }

  // make datagram queue
  XtcInput::DgramQueue dgqueue( 1 ) ;

  // start datagram reading thread
  boost::thread readerThread( XtcInput::DgramReader ( m_inputFiles.begin(), m_inputFiles.end(), dgqueue,
          m_mergeMode.value(), m_liveDbConn.value(), m_liveTable.value(), m_liveTimeout.value(), m_l1offset.value() ) ) ;

  // loop until there are events
  while ( true ) {
    
    XtcInput::Dgram dg = dgqueue.pop();
    if (dg.empty()) break;
    
    const Pds::Sequence& seq = dg.dg()->seq ;
    const Pds::ClockTime& clock = seq.clock() ;
    const Pds::TimeStamp& stamp = seq.stamp() ;
    printf("%s transition: damage %x, type %d, time %u sec %u nsec, ticks %u, fiducials %u, control %u, vector %u, payloadSize %d\n",
           TransitionId::name(seq.service()),
           dg.dg()->xtc.damage.value(),
           int(seq.type()),
           clock.seconds(), clock.nanoseconds(),
           stamp.ticks(),stamp.fiducials(),stamp.control(),stamp.vector(),
           dg.dg()->xtc.sizeofPayload());
    
    myLevelIter iter(&(dg.dg()->xtc));
    iter.iterate();
  }

  // stop reader thread if it is still running
  readerThread.interrupt();
  readerThread.join();

  return 0;
}

} // namespace O2OTranslator


// this defines main()
APPUTILS_MAIN(O2OTranslator::O2O_Scanner)
