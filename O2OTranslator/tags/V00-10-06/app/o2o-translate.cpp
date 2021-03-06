//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2O_Translate...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <vector>
#include <boost/thread/thread.hpp>
#include <sys/time.h>
#include <sys/resource.h>

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
#include "AppUtils/AppCmdOptIncr.h"
#include "AppUtils/AppCmdOptList.h"
#include "AppUtils/AppCmdOptSize.h"
#include "AppUtils/AppCmdOptNamedValue.h"
#include "LusiTime/Time.h"
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/MetaDataScanner.h"
#include "O2OTranslator/O2OFileNameFactory.h"
#include "O2OTranslator/O2OHdf5Writer.h"
#include "O2OTranslator/O2OMetaData.h"
#include "O2OTranslator/O2OXtcIterator.h"
#include "O2OTranslator/O2OXtcScannerI.h"
#include "O2OTranslator/O2OXtcValidator.h"
#include "XtcInput/DgramQueue.h"
#include "XtcInput/DgramReader.h"
#include "XtcInput/XtcFileName.h"
#include "XtcInput/XtcStreamMerger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

using namespace AppUtils ;

//
//  Application class
//
class O2O_Translate : public AppBase {
public:

  // Constructor
  explicit O2O_Translate ( const std::string& appName ) ;

  // destructor
  ~O2O_Translate () ;

protected :

  /**
   *  Main method which runs the whole application
   */
  virtual int runApp () ;

private:

  // more command line options and arguments
  AppCmdOptList<std::string>  m_optionsFile ;
  AppCmdOpt<std::string>      m_calibDir ;
  AppCmdOpt<int>              m_compression ;
  AppCmdOptSize               m_dgramsize ;
  AppCmdOpt<unsigned int>     m_dgramQSize ;
  AppCmdOpt<std::string>      m_experiment ;
  AppCmdOptBool               m_extGroups ;
  AppCmdOpt<std::string>      m_instrument ;
  AppCmdOpt<double>           m_l1offset ;
  AppCmdOpt<std::string>      m_mdConnStr ;
  AppCmdOptNamedValue<XtcInput::XtcStreamMerger::MergeMode> m_mergeMode ;
  AppCmdOptList<std::string>  m_metadata ;
  AppCmdOpt<std::string>      m_outputDir ;
  AppCmdOpt<std::string>      m_outputName ;
  AppCmdOptBool               m_overwrite ;
  AppCmdOpt<std::string>      m_regConnStr ;
  AppCmdOpt<unsigned long>    m_runNumber ;
  AppCmdOpt<std::string>      m_runType ;
  AppCmdOptNamedValue<O2OHdf5Writer::SplitMode> m_splitMode ;
  AppCmdOptSize               m_splitSize ;
  AppCmdArgList<std::string>  m_eventData ;
};

//----------------
// Constructors --
//----------------
O2O_Translate::O2O_Translate ( const std::string& appName )
  : AppBase( appName )
  , m_optionsFile( 'o', "options-file", "path",     "file name with options, multiple allowed", '\0' )
  , m_calibDir   ( 'C', "calib-dir",    "path",     "directory with calibration data, def: none", "" )
  , m_compression( 'c', "compression",  "number",   "compression level, -1..9, def: -1", -1 )
  , m_dgramsize  ( 'g', "datagram-size","size",     "datagram buffer size. def: 16M", 16*1048576ULL )
  , m_dgramQSize ( 'Q', "datagram-queue","number",  "datagram queue size. def: 32", 32 )
  , m_experiment ( 'x', "experiment",   "string",   "experiment name", "" )
  , m_extGroups  ( 'G', "group-time",               "use extended group names with timestamps", false )
  , m_instrument ( 'i', "instrument",   "string",   "instrument name", "" )
  , m_l1offset   (      "l1-offset",    "number",   "L1Accept time offset seconds, def: 0", 0 )
  , m_mdConnStr  ( 'M', "md-conn",      "string",   "metadata ODBC connection string", "" )
  , m_mergeMode  ( 'j', "merge-mode",   "mode-name","one of one-stream, no-chunking, file-name; def: file-name", 
                  XtcInput::XtcStreamMerger::FileName )
  , m_metadata   ( 'm', "metadata",     "name:value", "science metadata values", '\0' )
  , m_outputDir  ( 'd', "output-dir",   "path",     "directory to store output files, def: .", "." )
  , m_outputName ( 'n', "output-name",  "template", "template string for output file names, def: {seq4}.h5", "{seq4}.h5" )
  , m_overwrite  (      "overwrite",                "overwrite output file", false )
  , m_regConnStr ( 'R', "regdb-conn",   "string",   "RegDB ODBC connection string", "" )
  , m_runNumber  ( 'r', "run-number",   "number",   "run number, non-negative number; def: 0", 0 )
  , m_runType    ( 't', "run-type",     "string",   "run type, DATA or CALIB, def: DATA", "DATA" )
  , m_splitMode  ( 's', "split-mode",   "mode-name","one of none, or family; def: none", O2OHdf5Writer::NoSplit )
  , m_splitSize  ( 'z', "split-size",   "size",     "max. size of output files. def: 10G", 10*1073741824ULL )
  , m_eventData  ( "event-file",   "file name(s) with XTC event data" )
{
  setOptionsFile( m_optionsFile ) ;
  addOption( m_calibDir ) ;
  addOption( m_compression ) ;
  addOption( m_dgramsize ) ;
  addOption( m_dgramQSize ) ;
  addOption( m_experiment ) ;
  addOption( m_extGroups ) ;
  addOption( m_instrument ) ;
  addOption( m_l1offset ) ;
  addOption( m_mdConnStr ) ;
  addOption( m_mergeMode ) ;
  m_mergeMode.add ( "one-stream", XtcInput::XtcStreamMerger::OneStream ) ;
  m_mergeMode.add ( "no-chunking", XtcInput::XtcStreamMerger::NoChunking ) ;
  m_mergeMode.add ( "file-name", XtcInput::XtcStreamMerger::FileName ) ;
  addOption( m_metadata ) ;
  addOption( m_outputDir ) ;
  addOption( m_outputName ) ;
  addOption( m_overwrite ) ;
  addOption( m_regConnStr ) ;
  addOption( m_runNumber ) ;
  addOption( m_runType ) ;
  m_splitMode.add ( "none", O2OHdf5Writer::NoSplit ) ;
  m_splitMode.add ( "family", O2OHdf5Writer::Family ) ;
  addOption( m_splitMode ) ;
  addOption( m_splitSize ) ;
  addArgument( m_eventData ) ;
}

//--------------
// Destructor --
//--------------
O2O_Translate::~O2O_Translate ()
{
}

/**
 *  Main method which runs the whole application
 */
int
O2O_Translate::runApp ()
{
  LusiTime::Time start_time = LusiTime::Time::now() ;
  MsgLogRoot( info, "Starting translator process " << start_time ) ;
  MsgLogRoot( info, "Command line: " <<  this->cmdline() ) ;

  WithMsgLogRoot( info, log ) {
    typedef AppCmdOptList<std::string>::const_iterator Iter ;
    log << "input files:";
    for ( Iter it = m_eventData.begin() ; it != m_eventData.end() ; ++ it ) {
      log << "\n    " << *it ;
    }
  }
  MsgLogRoot( info, "output dir: " << m_outputDir.value() ) ;

  // instantiate name factory for output files
  O2OFileNameFactory nameFactory ( "{output-dir}/" + m_outputName.value() ) ;
  std::string outputDir = m_outputDir.value() ;
  if ( outputDir.empty() ) outputDir = "." ;
  nameFactory.addKeyword ( "output-dir", outputDir ) ;
  nameFactory.addKeyword ( "experiment", m_experiment.value() ) ;
  nameFactory.addKeyword ( "instrument", m_instrument.value() ) ;
  char runStr[16];
  snprintf ( runStr, sizeof runStr, "%06lu", m_runNumber.value() ) ;
  nameFactory.addKeyword ( "run", runStr ) ;

  // make metadata object
  O2OMetaData metadata ( m_runNumber.value(),
                         m_runType.value(),
                         m_instrument.value(),
                         m_experiment.value(),
                         m_calibDir.value(),
                         m_metadata.value() ) ;

  // instantiate XTC scanner, which is also output file writer
  std::vector<O2OXtcScannerI*> scanners ;
  scanners.push_back ( new O2OHdf5Writer ( nameFactory, m_overwrite.value(),
                                  m_splitMode.value(), m_splitSize.value(),
                                  m_compression.value(), m_extGroups.value(),
                                  metadata ) ) ;

  // instantiate metadata scanner
  scanners.push_back ( new MetaDataScanner( metadata, m_mdConnStr.value(), m_regConnStr.value() ) ) ;

  // make datagram queue
  XtcInput::DgramQueue dgqueue( m_dgramQSize.value() ) ;

  // start datagram reading thread
  std::list<XtcInput::XtcFileName> files ;
  for ( AppCmdOptList<std::string>::const_iterator it = m_eventData.begin() ; it != m_eventData.end() ; ++ it ) {
    files.push_back ( XtcInput::XtcFileName(*it) ) ;
  }
  boost::thread readerThread( XtcInput::DgramReader ( files, dgqueue, m_dgramsize.value(),
          m_mergeMode.value(), false, m_l1offset.value() ) ) ;

  uint64_t count = 0 ;

  // get all datagrams
  while ( true ) {
    
    XtcInput::Dgram dg = dgqueue.pop();
    if (dg.empty()) break;
    XtcInput::Dgram::ptr dgptr = dg.dg();

    ++ count ;

    WithMsgLogRoot( trace, out ) {
      const ClockTime& clock = dgptr->seq.clock() ;
      out << "Transition: #" << count << " "
          << std::left << std::setw(12) << Pds::TransitionId::name(dgptr->seq.service())
          << "  time: " << clock.seconds() << '.'
          << std::setfill('0') << std::setw(9) << clock.nanoseconds()
          << "  payloadSize: " << dgptr->xtc.sizeofPayload()
          << "  damage: " << std::hex << std::showbase << dgptr->xtc.damage.value() ;
    }

    // validate the XTC structure
    O2OXtcValidator validator ;
    if ( validator.process( &(dgptr->xtc) ) == 0 ) {

      WithMsgLogRoot( error, out ) {
        out << "Validation failed: Transition: #" << count << " "
            << Pds::TransitionId::name(dgptr->seq.service())
            << ", skipping datagram";
      }

    } else {

      // give this event to every scanner
      for ( std::vector<O2OXtcScannerI*>::iterator i = scanners.begin() ; i != scanners.end() ; ++ i ) {

        O2OXtcScannerI* scanner = *i ;

        try {
          if ( scanner->eventStart ( *dgptr ) ) {    
              O2OXtcIterator iter( &(dgptr->xtc), scanner );
              iter.iterate();
          }    
          scanner->eventEnd ( *dgptr ) ;
        } catch ( std::exception& e ) {
          MsgLogRoot( error, "exception caught processing datagram: " << e.what() ) ;
          return 3 ;
        }
      }

    }

  }

  // finish with the scanners
  for ( std::vector<O2OXtcScannerI*>::iterator i = scanners.begin() ; i != scanners.end() ; ++ i ) {
    try {
      O2OXtcScannerI* scanner = *i ;
      delete scanner ;
    } catch ( std::exception& e ) {
      MsgLogRoot( error, "exception caught while destroying a scanner: " << e.what() ) ;
      return 4 ;
    }
  }
  scanners.clear() ;

  LusiTime::Time end_time = LusiTime::Time::now() ;
  MsgLogRoot( info, "Translator process finished " << end_time ) ;

  // dump some resource info
  double delta_time = (end_time.sec()-start_time.sec()) + (end_time.nsec()-start_time.nsec()) / 1e9 ;
  WithMsgLogRoot( info, out ) {
    struct rusage u ;
    getrusage(RUSAGE_SELF, &u);
    out << "Resource usage summary:"
        << "\n    real time: " << delta_time
        << "\n    user time: " << (u.ru_utime.tv_sec + u.ru_utime.tv_usec/1e6)
        << "\n    sys time : " << (u.ru_stime.tv_sec + u.ru_stime.tv_usec/1e6) ;
  }

  return 0 ;

}

} // namespace O2OTranslator


// this defines main()
APPUTILS_MAIN(O2OTranslator::O2O_Translate)
