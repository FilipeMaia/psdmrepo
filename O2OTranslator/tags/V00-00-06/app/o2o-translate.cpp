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
#include "O2OTranslator/O2OFileNameFactory.h"
#include "O2OTranslator/O2OHdf5Writer.h"
#include "O2OTranslator/O2ONexusWriter.h"
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
  AppCmdOpt<std::string>      m_optionsFile ;
  AppCmdOptSize               m_dgramsize ;
  AppCmdOptList<std::string>  m_epicsData ;
  AppCmdOptList<std::string>  m_eventData ;
  AppCmdOpt<std::string>      m_experiment ;
  AppCmdOptBool               m_ignoreMiss ;
  AppCmdOptList<std::string>  m_metadata ;
  AppCmdOpt<std::string>      m_outputDir ;
  AppCmdOptBool               m_outputHdf5 ;
  AppCmdOpt<std::string>      m_outputName ;
  AppCmdOptBool               m_overwrite ;
  AppCmdOptNamedValue<O2OHdf5Writer::SplitMode> m_splitMode ;
  AppCmdOptSize               m_splitSize ;

};

//----------------
// Constructors --
//----------------
O2O_Translate::O2O_Translate ( const std::string& appName )
  : AppBase( appName )
  , m_optionsFile( 'o', "options-file", "path",     "file name with options", "" )
  , m_dgramsize  ( 'g', "datagram-size","size",     "datagram buffer size. def: 16M", 16*1048576ULL )
  , m_epicsData  ( 'e', "epics-file",   "path",     "file name for EPICS data", '\0' )
  , m_eventData  ( 'f', "event-file",   "path",     "file name for XTC event data", '\0' )
  , m_experiment ( 'x', "experiment",   "string",   "experiment name", "" )
  , m_ignoreMiss ( 'i', "ignore-miss",              "ignore missing XTC types, def: abort", false )
  , m_metadata   ( 'm', "metadata",     "name:value", "science metadata values", '\0' )
  , m_outputDir  ( 'd', "output-dir",   "path",     "directory to store output files, def: .", "." )
  , m_outputHdf5 ( '5', "output-hdf5",              "use HDF5 instead of NeXus for output file", false )
  , m_outputName ( 'n', "output-name",  "template", "template string for output file names, def: .", "{seq4}.hdf5" )
  , m_overwrite  (      "overwrite",                "overwrite output file", false )
  , m_splitMode  ( 's', "split-mode",   "mode-name","one of none, or family; def: none", O2OHdf5Writer::NoSplit )
  , m_splitSize  ( 'z', "split-size",   "number",   "max. size of output files. def: 20G", 20*1073741824ULL )
{
  setOptionsFile( m_optionsFile ) ;
  addOption( m_dgramsize ) ;
  addOption( m_epicsData ) ;
  addOption( m_eventData ) ;
  addOption( m_experiment ) ;
  addOption( m_ignoreMiss ) ;
  addOption( m_metadata ) ;
  addOption( m_outputDir ) ;
  addOption( m_outputHdf5 ) ;
  addOption( m_outputName ) ;
  addOption( m_overwrite ) ;
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

  // instantiate XTC scanner, which is also output file writer
  O2OXtcScannerI* scanner = 0 ;
  if ( m_outputHdf5.value() ) {
    scanner = new O2OHdf5Writer ( nameFactory, m_overwrite.value(), m_splitMode.value(), m_splitSize.value() ) ;
  } else {
    scanner = new O2ONexusWriter ( nameFactory ) ;
  }

  // loop over all input files
  typedef AppCmdOptList<std::string>::const_iterator StringIter ;
  StringIter eventFileIter = m_eventData.begin() ;
  StringIter epicsFileIter = m_epicsData.begin() ;
  while ( eventFileIter != m_eventData.end() ) {

    // get the file names
    const std::string& eventFile = *eventFileIter ;
    std::string epicsFile ;
    if ( epicsFileIter != m_epicsData.end() ) epicsFile = *epicsFileIter ;

    WithMsgLogRoot( info, log ) {
      log << "processing files:\n";
      log << "    " << eventFile ;
      if ( not epicsFile.empty() ) log << "\n    " << epicsFile ;
    }

    // open input xtc file
    FILE* xfile = fopen( eventFile.c_str(), "rb" );
    if ( ! xfile ) {
      MsgLogRoot( error, "failed to open input XTC file: " << eventFile ) ;
      return 2  ;
    }

    // open input EPICS file
    FILE* efile = 0 ;
    if ( not epicsFile.empty() ) {
      efile = fopen( epicsFile.c_str(), "rb" );
      if ( ! efile ) {
        MsgLogRoot( error, "failed to open input EPICS file: " << epicsFile ) ;
        return 2  ;
      }
    }

    // iterate over events in xtc file
    Pds::XtcFileIterator iter( xfile, m_dgramsize.value() ) ;
    while ( Pds::Dgram* dg = iter.next() ) {
      WithMsgLogRoot( trace, out ) {
        out << "Transition: "
            << std::left << std::setw(12) << Pds::TransitionId::name(dg->seq.service())
            << "  time: " << std::hex
            << std::showbase << std::internal << std::setfill('0')
            << std::setw(10) << dg->seq.high() << '/'
            << std::setw(10) << dg->seq.low()
            << "  payloadSize: " << std::dec << dg->xtc.sizeofPayload() ;
      }

      if ( scanner ) scanner->eventStart ( dg->seq ) ;

      O2OXtcIterator iter( &(dg->xtc), scanner, m_ignoreMiss.value() );
      iter.iterate();

      if ( scanner ) scanner->eventEnd ( dg->seq ) ;
    }

    // move to the next file
    ++ eventFileIter ;
    if ( epicsFileIter != m_epicsData.end() ) ++ epicsFileIter ;

  }

  // finish with the scanner
  delete scanner ;

  return 0 ;

}

} // namespace O2OTranslator


// this defines main()
APPUTILS_MAIN(O2OTranslator::O2O_Translate)
