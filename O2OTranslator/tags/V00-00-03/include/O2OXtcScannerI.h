#ifndef O2OTRANSLATOR_O2OXTCSCANNERI_H
#define O2OTRANSLATOR_O2OXTCSCANNERI_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OXtcScannerI.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "O2OTranslator/O2OPdsDataFwd.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Interface for a "visitor" class for XTC scanning.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @see AdditionalClass
 *
 *  @version $Id: template!C++!h 4 2008-10-08 19:27:36Z salnikov $
 *
 *  @author Andrei Salnikov
 */

namespace O2OTranslator {

class O2OXtcScannerI  {
public:

  // Destructor
  virtual ~O2OXtcScannerI () {}

  // signal start/end of the event
  virtual void eventStart ( const Pds::Sequence& seq ) = 0 ;
  virtual void eventEnd ( const Pds::Sequence& seq ) = 0 ;

  // signal start/end of the level
  virtual void levelStart ( const Pds::Src& src ) = 0 ;
  virtual void levelEnd ( const Pds::Src& src ) = 0 ;

  // visit the data object
  virtual void dataObject ( const Pds::WaveformV1& data, const Pds::Src& src ) = 0 ;
  virtual void dataObject ( const Pds::Acqiris::ConfigV1& data, const Pds::Src& src ) = 0 ;

protected:

  // Default constructor
  O2OXtcScannerI () {}

private:

  // Copy constructor and assignment are disabled by default
  O2OXtcScannerI ( const O2OXtcScannerI& ) ;
  O2OXtcScannerI& operator = ( const O2OXtcScannerI& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OXTCSCANNERI_H
