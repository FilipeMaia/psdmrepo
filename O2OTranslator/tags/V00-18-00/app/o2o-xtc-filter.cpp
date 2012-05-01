//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2O_XTC_Filter...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <boost/thread/thread.hpp>

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
#include "AppUtils/AppCmdOptList.h"
#include "AppUtils/AppCmdOptSize.h"
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/xtc/ClockTime.hh"
#include "pdsdata/xtc/TransitionId.hh"
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
class O2O_XTC_Filter : public AppBase {
public:

  // Constructor
  explicit O2O_XTC_Filter ( const std::string& appName ) ;

  // destructor
  ~O2O_XTC_Filter () ;

protected :

  /**
   *  Main method which runs the whole application
   */
  virtual int runApp () ;

private:

  // more command line options and arguments
  AppCmdOpt<int>              m_calibIndexOpt ;
  AppCmdOptList<int>          m_eventIndexOpt ;
  AppCmdOptSize               m_dgramsize ;
  AppCmdOpt<unsigned int>     m_dgramQSize ;
  AppCmdArgList<std::string>  m_inputFiles ;
  AppCmdArg<std::string>      m_outputName ;

  // other data members
  int m_val ;

};

//----------------
// Constructors --
//----------------
O2O_XTC_Filter::O2O_XTC_Filter ( const std::string& appName )
  : AppUtils::AppBase( appName )
  , m_calibIndexOpt( 'c', "calib-index", "number", "select event from given calib cycle, def: -1", -1 )
  , m_eventIndexOpt( 'e', "event-index", "number", "select event with given index, more than one possible" )
  , m_dgramsize  ( 'g', "datagram-size","size",     "datagram buffer size. def: 16M", 16*1048576ULL )
  , m_dgramQSize ( 'Q', "datagram-queue","number",  "datagram queue size. def: 32", 32 )
  , m_inputFiles ( "input-xtc", "the list of the input XTC files" )
  , m_outputName ( "output-xtc", "the name of the output XTC file" )
{
  addOption( m_calibIndexOpt ) ;
  addOption( m_eventIndexOpt ) ;
  addOption( m_dgramsize ) ;
  addOption( m_dgramQSize ) ;
  addArgument( m_inputFiles ) ;
  addArgument( m_outputName ) ;
}

//--------------
// Destructor --
//--------------
O2O_XTC_Filter::~O2O_XTC_Filter ()
{
}

/**
 *  Main method which runs the whole application
 */
int
O2O_XTC_Filter::runApp ()
{

  // open output file
  FILE* outstr = fopen ( m_outputName.value().c_str(), "wb" ) ;
  if ( outstr == 0 ) {
    MsgLogRoot( error, "failed to open output XTC file: " << m_outputName.value() ) ;
    return 2;
  }


  // make datagram queue
  XtcInput::DgramQueue dgqueue( m_dgramQSize.value() ) ;

  // start datagram reading thread
  std::list<XtcInput::XtcFileName> files ;
  for ( AppCmdArgList<std::string>::const_iterator it = m_inputFiles.begin() ; it != m_inputFiles.end() ; ++ it ) {
    files.push_back ( XtcInput::XtcFileName(*it) ) ;
  }
  boost::thread readerThread( XtcInput::DgramReader ( files, dgqueue, m_dgramsize.value(),
      XtcInput::XtcStreamMerger::FileName, false, 0 ) ) ;

  // seen transitions
  Pds::ClockTime transitions[Pds::TransitionId::NumberOf];

  int count = -1;
  int calibCycle = -1;
  bool passCalibCycle = false;
  int eventNumber = 0;

  // get all datagrams
  while ( true ) {

    XtcInput::Dgram dg = dgqueue.pop();
    if (dg.empty()) break;
    XtcInput::Dgram::ptr dgptr = dg.dg();

    // skip repeating transitions
    if (dgptr->seq.service() != Pds::TransitionId::L1Accept) {
      if (transitions[dgptr->seq.service()] == dgptr->seq.clock()) continue;
      transitions[dgptr->seq.service()] = dgptr->seq.clock();
    }

    ++ count ;

    WithMsgLogRoot( trace, out ) {
      const Pds::ClockTime& clock = dgptr->seq.clock() ;
      out << "Transition: #" << count << " "
          << std::left << std::setw(12) << Pds::TransitionId::name(dgptr->seq.service())
          << "  time: " << clock.seconds() << '.'
          << std::setfill('0') << std::setw(9) << clock.nanoseconds()
          << "  payloadSize: " << dgptr->xtc.sizeofPayload()
          << "  damage: " << std::hex << std::showbase << dgptr->xtc.damage.value() ;
    }

    bool pass = true;
    if (dgptr->seq.service() == Pds::TransitionId::BeginCalibCycle) {
      // new calib cycle
      ++ calibCycle;
      eventNumber = 0;
      passCalibCycle = m_calibIndexOpt.value() < 0 or m_calibIndexOpt.value() == calibCycle;
      pass = passCalibCycle;
    } else  if (dgptr->seq.service() == Pds::TransitionId::EndCalibCycle) {
      pass = passCalibCycle;
    } else  if (dgptr->seq.service() == Pds::TransitionId::L1Accept) {
      if (not passCalibCycle) {
        pass = false;
      } else if (not m_eventIndexOpt.empty()) {
        pass = std::find(m_eventIndexOpt.begin(), m_eventIndexOpt.end(), eventNumber) != m_eventIndexOpt.end();
      }
      ++ eventNumber;
    } else  if (dgptr->seq.service() == Pds::TransitionId::Enable) {
      pass = false;
    } else  if (dgptr->seq.service() == Pds::TransitionId::Disable) {
      pass = false;
    }

    MsgLogRoot(trace, "calibCycle=" << calibCycle << " event=" << (eventNumber-1) << " pass=" << (pass?"true":"false"));

    if (pass) {
      size_t dgsize = sizeof(Pds::Dgram) + dgptr->xtc.sizeofPayload() ;
      if ( fwrite ( dgptr.get(), dgsize, 1, outstr ) != 1 ) {
        MsgLogRoot( error, "failed to write datagram" ) ;
        return 3;
      }
    }
  }

  fclose(outstr);

  // return 0 on success, other values for error (like main())
  return 0 ;
}

} // namespace O2OTranslator


// this defines main()
APPUTILS_MAIN(O2OTranslator::O2O_XTC_Filter)
