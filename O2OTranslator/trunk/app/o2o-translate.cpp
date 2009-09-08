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
#include "Lusi/Lusi.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <vector>
#include <boost/thread/thread.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "AppUtils/AppBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdArg.h"
#include "AppUtils/AppCmdOpt.h"
#include "AppUtils/AppCmdOptBool.h"
#include "AppUtils/AppCmdOptIncr.h"
#include "AppUtils/AppCmdOptList.h"
#include "AppUtils/AppCmdOptSize.h"
#include "AppUtils/AppCmdOptNamedValue.h"
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/DgramQueue.h"
#include "O2OTranslator/DgramReader.h"
#include "O2OTranslator/MetaDataScanner.h"
#include "O2OTranslator/O2OFileNameFactory.h"
#include "O2OTranslator/O2OHdf5Writer.h"
#include "O2OTranslator/O2OMetaData.h"
#include "O2OTranslator/O2OXtcIterator.h"
#include "O2OTranslator/O2OXtcScannerI.h"
#include "pdsdata/xtc/XtcFileIterator.hh"

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
  AppCmdOpt<int>              m_compression ;
  AppCmdOptSize               m_dgramsize ;
  AppCmdOpt<unsigned int>     m_dgramQSize ;
  AppCmdOptList<std::string>  m_epicsData ;
  AppCmdOptList<std::string>  m_eventData ;
  AppCmdOpt<std::string>      m_experiment ;
  AppCmdOptBool               m_extGroups ;
  AppCmdOpt<std::string>      m_instrument ;
  AppCmdOpt<std::string>      m_mdConnStr ;
  AppCmdOptList<std::string>  m_metadata ;
  AppCmdOpt<std::string>      m_outputDir ;
  AppCmdOpt<std::string>      m_outputName ;
  AppCmdOptBool               m_overwrite ;
  AppCmdOpt<unsigned long>    m_runNumber ;
  AppCmdOpt<std::string>      m_runType ;
  AppCmdOptNamedValue<O2OHdf5Writer::SplitMode> m_splitMode ;
  AppCmdOptSize               m_splitSize ;

};

//----------------
// Constructors --
//----------------
O2O_Translate::O2O_Translate ( const std::string& appName )
  : AppBase( appName )
  , m_optionsFile( 'o', "options-file", "path",     "file name with options, multiple allowed", '\0' )
  , m_compression( 'c', "compression",  "number",   "compression level, -1..9, def: -1", -1 )
  , m_dgramsize  ( 'g', "datagram-size","size",     "datagram buffer size. def: 16M", 16*1048576ULL )
  , m_dgramQSize ( 'Q', "datagram-queue","number",     "datagram queue size. def: 32", 32 )
  , m_epicsData  ( 'e', "epics-file",   "path",     "file name for EPICS data", '\0' )
  , m_eventData  ( 'f', "event-file",   "path",     "file name for XTC event data", '\0' )
  , m_experiment ( 'x', "experiment",   "string",   "experiment name", "" )
  , m_extGroups  ( 'G', "group-time",               "use extended group names with timestamps", false )
  , m_instrument ( 'i', "instrument",   "string",   "instrument name", "" )
  , m_mdConnStr  ( 'M', "md-conn",      "string",   "metadata ODBC connection string", "" )
  , m_metadata   ( 'm', "metadata",     "name:value", "science metadata values", '\0' )
  , m_outputDir  ( 'd', "output-dir",   "path",     "directory to store output files, def: .", "." )
  , m_outputName ( 'n', "output-name",  "template", "template string for output file names, def: {seq4}.hdf5", "{seq4}.hdf5" )
  , m_overwrite  (      "overwrite",                "overwrite output file", false )
  , m_runNumber  ( 'r', "run-number",   "number",   "run number, non-negative number; def: 0", 0 )
  , m_runType    ( 't', "run-type",     "string",   "run type, DATA or CALIB, def: DATA", "DATA" )
  , m_splitMode  ( 's', "split-mode",   "mode-name","one of none, or family; def: family", O2OHdf5Writer::Family )
  , m_splitSize  ( 'z', "split-size",   "size",   "max. size of output files. def: 10G", 10*1073741824ULL )
{
  setOptionsFile( m_optionsFile ) ;
  addOption( m_compression ) ;
  addOption( m_dgramsize ) ;
  addOption( m_dgramQSize ) ;
  addOption( m_epicsData ) ;
  addOption( m_eventData ) ;
  addOption( m_experiment ) ;
  addOption( m_extGroups ) ;
  addOption( m_instrument ) ;
  addOption( m_mdConnStr ) ;
  addOption( m_metadata ) ;
  addOption( m_outputDir ) ;
  addOption( m_outputName ) ;
  addOption( m_overwrite ) ;
  addOption( m_runNumber ) ;
  addOption( m_runType ) ;
  m_splitMode.add ( "none", O2OHdf5Writer::NoSplit ) ;
  m_splitMode.add ( "family", O2OHdf5Writer::Family ) ;
  addOption( m_splitMode ) ;
  addOption( m_splitSize ) ;
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
  // verify input parameters, must ave at least one even data file and
  // if epics list is non-empty then its size must be equal to events
  if ( m_eventData.empty() ) {
    MsgLogRoot(error, "no event data files specified" ) ;
    return 2 ;
  }
  if ( not m_epicsData.empty() and m_eventData.size() != m_epicsData.size() ) {
    MsgLogRoot(error, "number of event/epics files is different" ) ;
    return 2 ;
  }

  WithMsgLogRoot( info, log ) {
    typedef AppCmdOptList<std::string>::const_iterator Iter ;
    log << "input files:";
    for ( Iter it = m_eventData.begin() ; it != m_eventData.end() ; ++ it ) {
      log << "\n    " << *it ;
    }
  }
  WithMsgLogRoot( info, log ) {
    typedef AppCmdOptList<std::string>::const_iterator Iter ;
    log << "epics files:";
    for ( Iter it = m_epicsData.begin() ; it != m_epicsData.end() ; ++ it ) {
      log << "\n    " << *it  ;
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
                         m_metadata.value() ) ;

  // instantiate XTC scanner, which is also output file writer
  std::vector<O2OXtcScannerI*> scanners ;
  scanners.push_back ( new O2OHdf5Writer ( nameFactory, m_overwrite.value(),
                                  m_splitMode.value(), m_splitSize.value(),
                                  m_compression.value(), m_extGroups.value(),
                                  metadata ) ) ;

  // instantiate metadata scanner
  scanners.push_back ( new MetaDataScanner( metadata, m_mdConnStr.value() ) ) ;

  // make datagram queue
  DgramQueue dgqueue( m_dgramQSize.value() ) ;

  // start datagram reading thread
  boost::thread readerThread( DgramReader ( m_eventData.value(), dgqueue, m_dgramsize.value() ) ) ;


  // get all datagrams
  while ( Pds::Dgram* dg = dgqueue.pop() ) {

    WithMsgLogRoot( trace, out ) {
      const ClockTime& clock = dg->seq.clock() ;
      out << "Transition: "
          << std::left << std::setw(12) << Pds::TransitionId::name(dg->seq.service())
          << "  time: " << clock.seconds() << '.'
          << std::setfill('0') << std::setw(9) << clock.nanoseconds()
          << "  payloadSize: " << dg->xtc.sizeofPayload() ;
    }

    // give this event to every scanner
    for ( std::vector<O2OXtcScannerI*>::iterator i = scanners.begin() ; i != scanners.end() ; ++ i ) {

      O2OXtcScannerI* scanner = *i ;

      try {
        scanner->eventStart ( *dg ) ;

        O2OXtcIterator iter( &(dg->xtc), scanner );
        iter.iterate();

        scanner->eventEnd ( *dg ) ;
      } catch ( std::exception& e ) {
        MsgLogRoot( error, "exception caught processing datagram: " << e.what() ) ;
        return 3 ;
      }
    }

    delete [] (char*)dg ;
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

  return 0 ;

}

} // namespace O2OTranslator


// this defines main()
APPUTILS_MAIN(O2OTranslator::O2O_Translate)
