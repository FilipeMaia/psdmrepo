//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2O_XTC_Repeat...
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
#include <fcntl.h>

//----------------------
// Base Class Headers --
//----------------------
#include "AppUtils/AppBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdArg.h"
#include "AppUtils/AppCmdArgList.h"
#include "AppUtils/AppCmdOpt.h"
#include "AppUtils/AppCmdOptBool.h"
#include "AppUtils/AppCmdOptIncr.h"
#include "AppUtils/AppCmdOptList.h"
#include "AppUtils/AppCmdOptSize.h"
#include "AppUtils/AppCmdOptNamedValue.h"
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OFileNameFactory.h"
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
class O2O_XTC_Repeat : public AppBase {
public:

  // Constructor
  explicit O2O_XTC_Repeat ( const std::string& appName ) ;

  // destructor
  ~O2O_XTC_Repeat () ;

protected :

  /**
   *  Main method which runs the whole application
   */
  virtual int runApp () ;

private:

  // more command line options and arguments
  AppCmdOpt<unsigned int>     m_count ;
  AppCmdOptSize               m_dgramsize ;
  AppCmdOptSize               m_minsize ;
  AppCmdOpt<std::string>      m_outputDir ;
  AppCmdOpt<std::string>      m_outputName ;
  AppCmdOptSize               m_splitSize ;
  AppCmdArgList<std::string>  m_inputFiles ;

  O2OFileNameFactory* m_nameFactory ;
  unsigned m_outSeq ;
  uint64_t m_outputSize ;
  FILE* m_out ;


  bool openNextFile() ;
  bool writeDatagram ( Pds::Dgram* dg ) ;
};

//----------------
// Constructors --
//----------------
O2O_XTC_Repeat::O2O_XTC_Repeat ( const std::string& appName )
  : AppBase( appName )
  , m_count      ( 'c', "count",        "number",   "repeat count for L1Accept datagrams, def: 1", 1 )
  , m_dgramsize  ( 'g', "datagram-size","size",     "datagram buffer size. def: 16M", 16*1048576ULL )
  , m_minsize    ( 'm', "min-size",     "size",     "minimum datagram size to copy. def: 0", 0 )
  , m_outputDir  ( 'd', "output-dir",   "path",     "directory to store output files, def: .", "." )
  , m_outputName ( 'n', "output-name",  "template", "template string for output file names, def: .", "{seq4}.xtc" )
  , m_splitSize  ( 'z', "split-size",   "number",   "max. size of output files. def: 2G", 2*1073741824ULL )
  , m_inputFiles ( "input-xtc", "the list of the input XTC files" )
  , m_outSeq(0)
  , m_outputSize(0)
  , m_out(0)
{
  addOption( m_count ) ;
  addOption( m_dgramsize ) ;
  addOption( m_minsize ) ;
  addOption( m_outputDir ) ;
  addOption( m_outputName ) ;
  addOption( m_splitSize ) ;
  addArgument( m_inputFiles ) ;
}

//--------------
// Destructor --
//--------------
O2O_XTC_Repeat::~O2O_XTC_Repeat ()
{
}

/**
 *  Main method which runs the whole application
 */
int
O2O_XTC_Repeat::runApp ()
{

  // instantiate name factory for output files
  m_nameFactory = new O2OFileNameFactory ( "{output-dir}/" + m_outputName.value() ) ;
  std::string outputDir = m_outputDir.value() ;
  if ( outputDir.empty() ) outputDir = "." ;
  m_nameFactory->addKeyword ( "output-dir", outputDir ) ;

  // output file
  if ( not openNextFile() ) {
    return 2  ;
  }

  // loop over all input files
  typedef AppCmdArgList<std::string>::const_iterator StringIter ;
  for ( StringIter fileIter = m_inputFiles.begin() ; fileIter != m_inputFiles.end() ; ++ fileIter ) {

    // get the file names
    const std::string& eventFile = *fileIter ;

    MsgLogRoot( info, "processing file: " << eventFile << '\n' ) ;

    // open input xtc file
    int xfile = open( eventFile.c_str(), O_RDONLY );
    if ( ! xfile ) {
      MsgLogRoot( error, "failed to open input XTC file: " << eventFile ) ;
      return 2  ;
    }

    // iterate over events in xtc file
    Pds::XtcFileIterator iter( xfile, m_dgramsize.value() ) ;
    while ( Pds::Dgram* dg = iter.next() ) {

      const Pds::Sequence& seq = dg->seq ;

      WithMsgLogRoot( trace, out ) {
        const Pds::ClockTime& clock = seq.clock() ;
        out << "Transition: "
            << std::left << std::setw(12) << Pds::TransitionId::name(dg->seq.service())
            << "  time: " << clock.seconds() << '.'
            << std::setfill('0') << std::setw(9) << clock.nanoseconds()
            << "  payloadSize: " << dg->xtc.sizeofPayload() ;
      }

      if ( seq.service() != Pds::TransitionId::L1Accept ) {

        // copy these to output file
        if ( not writeDatagram ( dg ) ) {
          return 2  ;
        }

      } else {

        size_t dgsize = sizeof(Pds::Dgram) + dg->xtc.sizeofPayload() ;
        if ( dgsize >= m_minsize.value() ) {

          // copy several instances to output file
          for ( unsigned i = 0 ; i < m_count.value() ; ++ i ) {
            if ( not writeDatagram ( dg ) ) {
              return 2  ;
            }
          }

        }

      }


    }

    close(xfile);

  }

  return 0 ;

}

bool
O2O_XTC_Repeat::openNextFile()
{
  // close open file
  if ( m_out ) fclose ( m_out ) ;
  m_outputSize = 0 ;

  // output file name
  std::string outPath = m_nameFactory->makePath( m_outSeq ) ;
  ++ m_outSeq ;

  // open it
  m_out = fopen ( outPath.c_str(), "wb" ) ;
  if ( m_out == 0 ) {
    MsgLogRoot( error, "failed to open output XTC file: " << outPath ) ;
  }
  return m_out != 0 ;
}

bool
O2O_XTC_Repeat::writeDatagram ( Pds::Dgram* dg )
{
  size_t dgsize = sizeof(Pds::Dgram) + dg->xtc.sizeofPayload() ;

  if ( m_outputSize > 0 and (m_outputSize+dgsize) > m_splitSize.value() ) {
    if ( not openNextFile() ) return false ;
  }

  if ( fwrite ( dg, dgsize, 1, m_out ) != 1 ) {
    MsgLogRoot( error, "failed to write datagram" ) ;
    return false ;
  }

  m_outputSize += dgsize ;
  return true ;
}

} // namespace O2OTranslator


// this defines main()
APPUTILS_MAIN(O2OTranslator::O2O_XTC_Repeat)
