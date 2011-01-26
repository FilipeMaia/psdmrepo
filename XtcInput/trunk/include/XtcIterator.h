#ifndef XTCINPUT_XTCITERATOR_H
#define XTCINPUT_XTCITERATOR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcIterator.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "pdsdata/xtc/XtcIterator.hh"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "XtcInput/XtcSrcStack.h"
#include "pdsdata/xtc/TypeId.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Subclass of XtcIterator which forwards all
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

class XtcScannerI ;

class XtcIterator : public Pds::XtcIterator {
public:

  // Default constructor
  XtcIterator ( Xtc* xtc, XtcScannerI* scanner ) ;

  // Destructor
  virtual ~XtcIterator () ;

  // process one sub-XTC, returns >0 for success, 0 for error
  virtual int process(Xtc* xtc) ;

protected:

private:

  // Data members
  XtcScannerI* m_scanner ;
  XtcSrcStack m_src ;

  // Copy constructor and assignment are disabled by default
  XtcIterator ( const XtcIterator& ) ;
  XtcIterator operator = ( const XtcIterator& ) ;

};

} // namespace XtcInput

#endif // XTCINPUT_XTCITERATOR_H
