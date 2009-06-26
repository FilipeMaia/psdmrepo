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
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/MetaDataScanner.h"

//-----------------
// C/C++ Headers --
//-----------------
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
                                  const std::string& odbcConnStr)
  : O2OXtcScannerI()
  , m_metadata(metadata)
  , m_odbcConnStr(odbcConnStr)
  , m_nevents(0)
  , m_eventSize(0)
  , m_runBeginTime()
  , m_runEndTime()
{
}

//--------------
// Destructor --
//--------------
MetaDataScanner::~MetaDataScanner ()
{
}

// signal start/end of the event (datagram)
void
MetaDataScanner::eventStart ( const Pds::Dgram& dgram )
{
  if ( dgram.seq.service() == Pds::TransitionId::Map ) {


  } else if ( dgram.seq.service() == Pds::TransitionId::Configure ) {


  } else if ( dgram.seq.service() == Pds::TransitionId::BeginRun ) {

    // reset run-specific stats
    resetRunInfo() ;

    const Pds::ClockTime& t = dgram.seq.clock() ;
    m_runBeginTime = LusiTime::Time ( t.seconds(), t.nanoseconds() ) ;

  } else if ( dgram.seq.service() == Pds::TransitionId::L1Accept ) {

    // increment counters
    ++ m_nevents ;
    m_eventSize += dgram.xtc.extent ;

  } else if ( dgram.seq.service() == Pds::TransitionId::EndRun ) {

    const Pds::ClockTime& t = dgram.seq.clock() ;
    m_runEndTime = LusiTime::Time ( t.seconds(), t.nanoseconds() ) ;

    // store run-specific stats
    storeRunInfo() ;

  } else if ( dgram.seq.service() == Pds::TransitionId::Unconfigure ) {


  } else if ( dgram.seq.service() == Pds::TransitionId::Unmap ) {


  }
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
MetaDataScanner::dataObject ( const void* data, const Pds::TypeId& typeId, const Pds::DetInfo& detInfo )
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
  MsgLog( logger, info, "run statistics:"
      << "\n\tm_runNumber: " << m_metadata.runNumber()
      << "\n\tm_experiment: " << m_metadata.experiment()
      << "\n\tm_nevents: " << m_nevents
      << "\n\tm_eventSize: " << ( m_nevents ? m_eventSize/m_nevents : 0 )
      << "\n\tm_runBeginTime: " << m_runBeginTime.toString("S%s%f") << "(" << m_runBeginTime << ")"
      << "\n\tm_runEndTime: " << m_runEndTime.toString("S%s%f") << "(" << m_runEndTime << ")" ) ;

  // can we store it?
  if ( m_odbcConnStr.empty() ) {
    MsgLog( logger, warning, "metadata ODBC connection string is empty, no data will be stored" ) ;
    return ;
  }
  if ( m_metadata.runNumber() == 0 ) {
    MsgLog( logger, warning, "run number is zero, no data will be stored" ) ;
    return ;
  }
  if ( m_metadata.experiment().empty() ) {
    MsgLog( logger, warning, "experiment name is empty, no data will be stored" ) ;
    return ;
  }

  // create connection to the SciMD
  SciMD::Connection* conn = 0 ;
  try {
    MsgLog( logger, trace, "ODBC connection string: '" << m_odbcConnStr << "'" ) ;
    conn = SciMD::Connection::open( m_odbcConnStr ) ;
  } catch ( SciMD::DatabaseError& e ) {
    MsgLog( logger, error, "failed to open SciMD connection, metadata will not be stored" ) ;
    throw ;
  }

  // start saving data
  conn->beginTransaction() ;

  // create info for this run
  try {
    conn->createRun ( m_metadata.experiment(), m_metadata.runNumber(), m_metadata.runType(), m_runBeginTime, m_runEndTime ) ;
  } catch ( SciMD::DatabaseError& e ) {
    MsgLog( logger, error, "failed to create new run, run number may already exist" ) ;
    conn->abortTransaction() ;
    throw ;
  }

  try {

    // store event count
    conn->setRunParam ( m_metadata.experiment(), m_metadata.runNumber(),
                        "events", (int)m_nevents, "translator" ) ;

  } catch ( std::exception& e ) {

    // this should not happen, have to abort here
    MsgLog( logger, error, "failed to store event count: " << e.what() ) ;
    conn->abortTransaction() ;
    throw ;

  }

  try {

    // store average event size
    conn->setRunParam ( m_metadata.experiment(), m_metadata.runNumber(),
                        "eventSize", (int)( m_nevents ? m_eventSize/m_nevents : 0 ), "translator" ) ;

  } catch ( std::exception& e ) {

    // this should not happen, have to abort here
    MsgLog( logger, error, "failed to store event size: " << e.what() ) ;
    conn->abortTransaction() ;
    throw ;

  }

  // store all metadata that we received from online
  typedef O2OMetaData::const_iterator MDIter ;
  for ( MDIter it = m_metadata.extra_begin() ; it != m_metadata.extra_end() ; ++ it ) {
    try {
      conn->setRunParam ( m_metadata.experiment(), m_metadata.runNumber(), it->first, it->second, "translator" ) ;
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
