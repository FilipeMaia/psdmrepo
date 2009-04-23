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
#include "pdsdata/xtc/Sequence.hh"
#include "pdsdata/xtc/Src.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
MetaDataScanner::MetaDataScanner ()
  : O2OXtcScannerI()
  , m_nevents(0)
  , m_runBeginTime(0,0)
  , m_runEndTime(0,0)
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
MetaDataScanner::eventStart ( const Pds::Sequence& seq )
{
  if ( seq.service() == Pds::TransitionId::Map ) {


  } else if ( seq.service() == Pds::TransitionId::Configure ) {


  } else if ( seq.service() == Pds::TransitionId::BeginRun ) {

    // reset run-specific stats
    resetRunInfo() ;

    m_runBeginTime = seq.clock() ;

  } else if ( seq.service() == Pds::TransitionId::L1Accept ) {

    // increment counters
    ++ m_nevents ;

  } else if ( seq.service() == Pds::TransitionId::EndRun ) {

    m_runEndTime = seq.clock() ;

    // store run-specific stats
    storeRunInfo() ;

  } else if ( seq.service() == Pds::TransitionId::Unconfigure ) {


  } else if ( seq.service() == Pds::TransitionId::Unmap ) {


  }
}

void
MetaDataScanner::eventEnd ( const Pds::Sequence& seq )
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
  m_runEndTime = m_runBeginTime = Pds::ClockTime(0,0) ;
}

// store collected run statistics
void
MetaDataScanner::storeRunInfo()
{
  std::cout << "m_nevents: " << m_nevents << '\n' ;
  std::cout << "m_runBeginTime: " << m_runBeginTime.seconds() << '.'
      << std::setfill('0') << std::setw(9) << m_runBeginTime.nanoseconds() << '\n' ;
  std::cout << "m_runEndTime: " << m_runEndTime.seconds() << '.'
      << std::setfill('0') << std::setw(9) << m_runEndTime.nanoseconds() << '\n' ;
}

} // namespace O2OTranslator
