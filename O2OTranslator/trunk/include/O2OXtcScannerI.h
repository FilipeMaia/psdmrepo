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
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "O2OTranslator/O2OXtcSrc.h"
#include "pdsdata/xtc/Damage.hh"
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/TypeId.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------


namespace O2OTranslator {

/**
 *  @brief Interface for a "visitor" class for XTC scanning.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class O2OXtcScannerI : boost::noncopyable {
public:

  // Destructor
  virtual ~O2OXtcScannerI () {}

  // signal start/end of the event (datagram), if eventStart returns
  // false then the datagram should be discarded
  virtual bool eventStart ( const Pds::Dgram& dgram ) = 0 ;
  virtual void eventEnd ( const Pds::Dgram& dgram ) = 0 ;

  // signal start/end of the level
  virtual void levelStart ( const Pds::Src& src ) = 0 ;
  virtual void levelEnd ( const Pds::Src& src ) = 0 ;

  // visit the data object
  virtual void dataObject ( const void* data, 
                            size_t size, 
                            const Pds::TypeId& typeId, 
                            const O2OXtcSrc& src,
                            Pds::Damage damage ) = 0 ;

  // visit the data object in configure or begincalibcycle transitions
  virtual void configObject(const void* data,
                            size_t size,
                            const Pds::TypeId& typeId,
                            const O2OXtcSrc& src,
                            Pds::Damage damage) = 0;

protected:

  // Default constructor
  O2OXtcScannerI () {}

private:

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OXTCSCANNERI_H
