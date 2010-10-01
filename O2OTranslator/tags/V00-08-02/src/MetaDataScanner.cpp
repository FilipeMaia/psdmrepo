//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MetaDataScanner...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/MetaDataScanner.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <iostream>
#include <iomanip>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OMetaData.h"
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/Src.hh"
#include "SciMD/Connection.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "MDScanner" ;

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
MetaDataScanner::MetaDataScanner (const O2OMetaData& metadata,
                                  const std::string& odbcConnStr,
                                  const std::string& regdbConnStr)
  : O2OXtcScannerI()
  , m_metadata(metadata)
  , m_odbcConnStr(odbcConnStr)
  , m_regdbConnStr(regdbConnStr)
  , m_nevents(0)
  , m_eventSize(0)
  , m_runBeginTime()
  , m_runEndTime()
  , m_stored(false)
  , m_transClock()
{
  std::fill_n(m_transClock, int(Pds::TransitionId::NumberOf), LusiTime::Time(0,0));
}

//--------------
// Destructor --
//--------------
MetaDataScanner::~MetaDataScanner ()
{
  if ( not m_stored ) {

    // in case the EndRun transition is not seen just dump
    // the collected info to the log file

    const std::string& instr = m_metadata.instrument() ;
    const std::string& exper = m_metadata.experiment() ;
    unsigned long run = m_metadata.runNumber() ;

    MsgLog( logger, info, "run statistics:"
        << "\n\tm_runNumber: " << run
        << "\n\tm_instrument: " << instr
        << "\n\tm_experiment: " << exper
        << "\n\tm_nevents: " << m_nevents
        << "\n\tm_eventSize: " << ( m_nevents ? m_eventSize/m_nevents : 0 )
        << "\n\tm_runBeginTime: " << m_runBeginTime.toString("S%s%f") << "(" << m_runBeginTime << ")"
        << "\n\tm_runEndTime: " << m_runEndTime.toString("S%s%f") << "(" << m_runEndTime << ")" ) ;

  }

}

// signal start/end of the event (datagram)
bool
MetaDataScanner::eventStart ( const Pds::Dgram& dgram )
{
  Pds::TransitionId::Value transition = dgram.seq.service();
  Pds::ClockTime clock = dgram.seq.clock();
  LusiTime::Time t(clock.seconds(), clock.nanoseconds());

  if ( dgram.seq.service() == Pds::TransitionId::Map ) {


  } else if ( dgram.seq.service() == Pds::TransitionId::Configure ) {


  } else if ( dgram.seq.service() == Pds::TransitionId::BeginRun ) {

    if ( t != m_transClock[transition] ) {
      // reset run-specific stats
      resetRunInfo() ;
  
      m_runBeginTime = t ;
    }

  } else if ( dgram.seq.service() == Pds::TransitionId::L1Accept ) {

    // check the time, should not be sooner than begin of calib cycle
    if ( t >= m_transClock[Pds::TransitionId::BeginCalibCycle] ) {
      // increment counters
      ++ m_nevents ;
      m_eventSize += dgram.xtc.extent ;
    }

  } else if ( dgram.seq.service() == Pds::TransitionId::EndRun ) {

    if ( t != m_transClock[transition] ) {
      const Pds::ClockTime& t = dgram.seq.clock() ;
      m_runEndTime = LusiTime::Time ( t.seconds(), t.nanoseconds() ) ;
  
      // store run-specific stats
      storeRunInfo() ;
      m_stored = true ;
    }

  } else if ( dgram.seq.service() == Pds::TransitionId::Unconfigure ) {


  } else if ( dgram.seq.service() == Pds::TransitionId::Unmap ) {


  }
  
  // store the time of the transition
  m_transClock[transition] = t;
  
  return true;
}

void
MetaDataScanner::eventEnd ( const Pds::Dgram& dgram )
{
}

// signal start/end of the level
void
MetaDataScanner::levelStart ( const Pds::Src& src )
{

}

void
MetaDataScanner::levelEnd ( const Pds::Src& src )
{

}

// visit the data object
void
MetaDataScanner::dataObject ( const void* data, size_t size,
    const Pds::TypeId& typeId, const O2OXtcSrc& src )
{

}

// reset the run statistics
void
MetaDataScanner::resetRunInfo()
{
  m_nevents = 0 ;
  m_runEndTime = m_runBeginTime = LusiTime::Time(0,0) ;
}

// store collected run statistics
void
MetaDataScanner::storeRunInfo()
{
  const std::string& instr = m_metadata.instrument() ;
  const std::string& exper = m_metadata.experiment() ;
  unsigned long run = m_metadata.runNumber() ;

  MsgLog( logger, info, "run statistics:"
      << "\n\tm_runNumber: " << run
      << "\n\tm_instrument: " << instr
      << "\n\tm_experiment: " << exper
      << "\n\tm_nevents: " << m_nevents
      << "\n\tm_eventSize: " << ( m_nevents ? m_eventSize/m_nevents : 0 )
      << "\n\tm_runBeginTime: " << m_runBeginTime.toString("S%s%f") << "(" << m_runBeginTime << ")"
      << "\n\tm_runEndTime: " << m_runEndTime.toString("S%s%f") << "(" << m_runEndTime << ")" ) ;

  // can we store it?
  if ( m_odbcConnStr.empty() ) {
    MsgLog( logger, warning, "metadata ODBC connection string is empty, no data will be stored" ) ;
    return ;
  }
  if ( run == 0 ) {
    MsgLog( logger, warning, "run number is zero, no data will be stored" ) ;
    return ;
  }
  if ( instr.empty() ) {
    MsgLog( logger, warning, "instrument name is empty, no data will be stored" ) ;
    return ;
  }
  if ( exper.empty() ) {
    MsgLog( logger, warning, "experiment name is empty, no data will be stored" ) ;
    return ;
  }

  // create connection to the SciMD
  SciMD::Connection* conn = 0 ;
  try {
    MsgLog( logger, trace, "ODBC connection string: '" << m_odbcConnStr << "'" ) ;
    conn = SciMD::Connection::open( m_odbcConnStr, m_regdbConnStr ) ;
  } catch ( SciMD::DatabaseError& e ) {
    MsgLog( logger, error, "failed to open SciMD connection, metadata will not be stored" ) ;
    throw ;
  }

  // start saving data
  conn->beginTransaction() ;

  // create info for this run
  try {
    conn->createRun ( instr, exper, run, m_metadata.runType(), m_runBeginTime, m_runEndTime ) ;
  } catch ( SciMD::DatabaseError& e ) {
    MsgLog( logger, error, "failed to create new run, run number may already exist" ) ;
    conn->abortTransaction() ;
    delete conn ;
    throw ;
  }

  try {

    // store event count
    conn->setRunParam ( instr, exper, run, "events", (int)m_nevents, "translator" ) ;

  } catch ( std::exception& e ) {

    // this should not happen, have to abort here
    MsgLog( logger, error, "failed to store event count: " << e.what() ) ;
    conn->abortTransaction() ;
    delete conn ;
    throw ;

  }

  try {

    // store average event size
    conn->setRunParam ( instr, exper, run,
                        "eventSize", (int)( m_nevents ? m_eventSize/m_nevents : 0 ), "translator" ) ;

  } catch ( std::exception& e ) {

    // this should not happen, have to abort here
    MsgLog( logger, error, "failed to store event size: " << e.what() ) ;
    conn->abortTransaction() ;
    delete conn ;
    throw ;

  }

  try {

    // store average event size
    conn->setRunParam ( instr, exper, run, "dgramSize", (int64_t)m_eventSize, "translator" ) ;

  } catch ( std::exception& e ) {

    // this should not happen, have to abort here
    MsgLog( logger, error, "failed to store datagrams size: " << e.what() ) ;
    conn->abortTransaction() ;
    delete conn ;
    throw ;

  }

  // store all metadata that we received from online
  typedef O2OMetaData::const_iterator MDIter ;
  for ( MDIter it = m_metadata.extra_begin() ; it != m_metadata.extra_end() ; ++ it ) {
    try {
      conn->setRunParam ( instr, exper, run, it->first, it->second, "translator" ) ;
    } catch ( std::exception& e ) {
      // this is not fatal, just print error message and continue
      MsgLog( logger, error, "failed to store metadata: " << e.what()
          << "\n\tkey='" << it->first << "', value='" << it->second << "'" ) ;
    }
  }

  // finished storing metadata
  conn->commitTransaction() ;
  delete conn ;
  conn = 0 ;
}

} // namespace O2OTranslator
