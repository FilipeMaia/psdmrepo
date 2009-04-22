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

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/O2OXtcScannerI.h"
#include "pdsdata/xtc/ClockTime.hh"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
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
  MetaDataScanner () ;

  // Destructor
  virtual ~MetaDataScanner () ;

  // signal start/end of the event (datagram)
  virtual void eventStart ( const Pds::Sequence& seq ) ;
  virtual void eventEnd ( const Pds::Sequence& seq ) ;

  // signal start/end of the level
  virtual void levelStart ( const Pds::Src& src ) ;
  virtual void levelEnd ( const Pds::Src& src ) ;

  // visit the data object
  virtual void dataObject ( const void* data, const Pds::TypeId& typeId, const Pds::DetInfo& detInfo ) ;

protected:

  // reset the run statistics
  void resetRunInfo() ;

  // store collected run statistics
  void storeRunInfo() ;

private:

  // Data members
  unsigned long m_nevents ;
  Pds::ClockTime m_runBeginTime ;
  Pds::ClockTime m_runEndTime ;

  // Copy constructor and assignment are disabled by default
  MetaDataScanner ( const MetaDataScanner& ) ;
  MetaDataScanner& operator = ( const MetaDataScanner& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_METADATASCANNER_H
