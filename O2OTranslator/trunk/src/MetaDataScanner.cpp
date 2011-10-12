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
MetaDataScanner::MetaDataScanner (const O2OMetaData& metadata)
  : O2OXtcScannerI()
  , m_metadata(metadata)
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
}

} // namespace O2OTranslator
