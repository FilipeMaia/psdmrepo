/**
 * Original code for XTC scanning from Chris
 *
 * $Id$
 */
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <list>

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
#include "O2OTranslator/O2OXtcFileName.h"
#include "O2OTranslator/O2OXtcMerger.h"
#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/ProcInfo.hh"
#include "pdsdata/xtc/XtcIterator.hh"
#include "pdsdata/xtc/XtcFileIterator.hh"



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
    printf("%s level: ",Level::name(level));
    printf("payloadSize %d ", xtc->sizeofPayload() );
    printf("damage %x ", xtc->damage.value() );
    if (level==Level::Source or level==Pds::Level::Reporter or level==Pds::Level::Control) {
      DetInfo& info = *(DetInfo*)(&xtc->src);
      printf("%s.%d %s.%d",
             DetInfo::name(info.detector()),info.detId(),
             DetInfo::name(info.device()),info.devId());
    } else {
      ProcInfo& info = *(ProcInfo*)(&xtc->src);
      printf("IpAddress %#x ProcessId %#x",info.ipAddr(),info.processId());
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
  AppCmdOptNamedValue<O2OXtcMerger::MergeMode> m_mergeMode ;
  AppCmdArgList<std::string>  m_inputFiles ;

};

//----------------
// Constructors --
//----------------
O2O_Scanner::O2O_Scanner ( const std::string& appName )
  : AppBase( appName )
  , m_skipDamaged( 'd', "skip-damaged",             "skip damaged datagrams", false )
  , m_l1offset   (      "l1-offset",    "number",   "L1Accept time offset seconds, def: 0", 0 )
  , m_mergeMode  ( 'j', "merge-mode",   "mode-name","one of one-stream, no-chunking, file-name; def: file-name", O2OXtcMerger::FileName )
  , m_inputFiles ( "input-xtc", "the list of the input XTC files" )
{
  addOption( m_skipDamaged ) ;
  addOption( m_l1offset ) ;
  addOption( m_mergeMode ) ;
  m_mergeMode.add ( "one-stream", O2OXtcMerger::OneStream ) ;
  m_mergeMode.add ( "no-chunking", O2OXtcMerger::NoChunking ) ;
  m_mergeMode.add ( "file-name", O2OXtcMerger::FileName ) ;
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
  std::list<O2OXtcFileName> files ;
  for (AppCmdArgList<std::string>::const_iterator i = m_inputFiles.begin(); i!=m_inputFiles.end(); ++i) {
    files.push_back( O2OXtcFileName(*i) ) ;
  }

  if (files.empty()) {
    MsgLogRoot(error, "no input data files specified" ) ;
    return 2 ;
  }

  O2OTranslator::O2OXtcMerger iter(files, 0x1000000, m_mergeMode.value(), m_skipDamaged.value(), m_l1offset.value());
  while ( Dgram* dg = iter.next() ) {
    const Pds::Sequence& seq = dg->seq ;
    const Pds::ClockTime& clock = seq.clock() ;
    const Pds::TimeStamp& stamp = seq.stamp() ;
    printf("%s transition: damage %x, type %d, time %u sec %u nsec, ticks %u, fiducials %u, control %u, payloadSize %d\n",
           TransitionId::name(seq.service()),
           dg->xtc.damage.value(),
           int(seq.type()),
           clock.seconds(), clock.nanoseconds(),
           stamp.ticks(),stamp.fiducials(),stamp.control(),
           dg->xtc.sizeofPayload());
    myLevelIter iter(&(dg->xtc));
    iter.iterate();

    delete [] (char*)dg ;
  }

  return 0;
}

} // namespace O2OTranslator


// this defines main()
APPUTILS_MAIN(O2OTranslator::O2O_Scanner)
