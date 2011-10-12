#ifndef O2OTRANSLATOR_METADATASCANNER_H
#define O2OTRANSLATOR_METADATASCANNER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MetaDataScanner.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <stdint.h>
#include <string>
#include <list>

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/O2OXtcScannerI.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "LusiTime/Time.h"
#include "pdsdata/xtc/ClockTime.hh"
#include "pdsdata/xtc/TransitionId.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

class O2OMetaData ;

/**
 *  XTC scanner class which collect metadata and stores them in SciMD database.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class MetaDataScanner : public O2OXtcScannerI {
public:

  // Default constructor
  MetaDataScanner ( const O2OMetaData& metadata ) ;

  // Destructor
  virtual ~MetaDataScanner () ;

  // signal start/end of the event (datagram)
  virtual bool eventStart ( const Pds::Dgram& dgram ) ;
  virtual void eventEnd ( const Pds::Dgram& dgram ) ;

  // signal start/end of the level
  virtual void levelStart ( const Pds::Src& src ) ;
  virtual void levelEnd ( const Pds::Src& src ) ;

  // visit the data object
  virtual void dataObject ( const void* data, size_t size,
       const Pds::TypeId& typeId, const O2OXtcSrc& src ) ;

protected:

  // reset the run statistics
  void resetRunInfo() ;

  // store collected run statistics
  void storeRunInfo() ;

private:

  typedef LusiTime::Time TransitionClock[Pds::TransitionId::NumberOf] ;

  // Data members
  const O2OMetaData& m_metadata ;
  unsigned long m_nevents ;
  uint64_t m_eventSize ;
  LusiTime::Time m_runBeginTime ;
  LusiTime::Time m_runEndTime ;
  bool m_stored ;
  TransitionClock m_transClock;

  // Copy constructor and assignment are disabled by default
  MetaDataScanner ( const MetaDataScanner& ) ;
  MetaDataScanner& operator = ( const MetaDataScanner& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_METADATASCANNER_H
