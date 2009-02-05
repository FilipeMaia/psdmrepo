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
#include "AppUtils/AppCmdOptIncr.h"
#include "MsgLogger/MsgLogger.h"
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

//
//  Application class
//
class O2O_Translate : public AppUtils::AppBase {
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
  AppUtils::AppCmdOpt<unsigned int> m_dgramsize ;
  AppUtils::AppCmdOpt<std::string> m_epics ;
  AppUtils::AppCmdOptIncr m_ignore ;
  AppUtils::AppCmdOptIncr m_hdf5 ;
  AppUtils::AppCmdArg<std::string> m_input ;
  AppUtils::AppCmdArg<std::string> m_output ;

};

//----------------
// Constructors --
//----------------
O2O_Translate::O2O_Translate ( const std::string& appName )
  : AppUtils::AppBase( appName )
  , m_dgramsize( 'd', "dgram-size", "number", "datagram buffer size, MB. def: 16", 16 )
  , m_epics( 'e', "epics-file", "path", "file name for EPICS data", "" )
  , m_ignore( 'i', "ignore", "ignore unknown xtc types, def: abort", 0 )
  , m_hdf5( '5', "hdf5", "use hdf5 instead of nexus for output file", 0 )
  , m_input( "input-file", "input file name" )
  , m_output( "output-file", "output file name" )
{
  addOption( m_dgramsize ) ;
  addOption( m_epics ) ;
  addOption( m_ignore ) ;
  addOption( m_hdf5 ) ;
  addArgument( m_input ) ;
  addArgument( m_output ) ;
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
try {

  MsgLogRoot( info, "input file name: " << m_input.value() ) ;
  MsgLogRoot( info, "output file name: " << m_output.value() ) ;

  // open input xtc file
  FILE* xfile = fopen( m_input.value().c_str(), "rb" );
  if ( ! xfile ) {
    MsgLogRoot( error, "failed to open input XTC file: " << m_output.value() ) ;
    return 2  ;
  }

  // open input EPICS file
  FILE* efile = 0 ;
  if ( not m_epics.value().empty() ) {
    efile = fopen( m_epics.value().c_str(), "rb" );
    if ( ! efile ) {
      MsgLogRoot( error, "failed to open input EPICS file: " << m_epics.value() ) ;
      return 2  ;
    }
  }

  // instantiate XTC scanner, which is also output file writer
  O2OXtcScannerI* scanner = 0 ;
  if ( not m_hdf5.value() ) {
    scanner = new O2ONexusWriter ( m_output.value() ) ;
  }

  // iterate over events in xtc file
  Pds::XtcFileIterator iter( xfile, m_dgramsize.value()*1048576 ) ;
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

    O2OXtcIterator iter( &(dg->xtc), scanner, m_ignore.value() );
    iter.iterate();

    if ( scanner ) scanner->eventEnd ( dg->seq ) ;
  }

  // this will close the file
  delete scanner ;

  return 0 ;

} catch ( const std::exception& e ) {

  MsgLogRoot( error, "exception caught: " << e.what() ) ;
  return 2 ;

} catch ( ... ) {

  MsgLogRoot( error, "unknown exception caught" ) ;
  return 2 ;

}

} // namespace O2OTranslator


// this defines main()
APPUTILS_MAIN(O2OTranslator::O2O_Translate)
