#ifndef XTCINPUT_XTCSCANNERI_H
#define XTCINPUT_XTCSCANNERI_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcScannerI.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/Src.hh"
#include "pdsdata/xtc/TypeId.hh"
#include "XtcInput/XtcSrcStack.h"

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
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace XtcInput {

class XtcScannerI : boost::noncopyable {
public:

  // Destructor
  virtual ~XtcScannerI () {}

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
                            const XtcSrcStack& src ) = 0 ;

protected:

  // Default constructor
  XtcScannerI () {}

private:

  // Copy constructor and assignment are disabled by default
  XtcScannerI ( const XtcScannerI& ) ;
  XtcScannerI& operator = ( const XtcScannerI& ) ;

};

} // namespace XtcInput

#endif // XTCINPUT_XTCSCANNERI_H
