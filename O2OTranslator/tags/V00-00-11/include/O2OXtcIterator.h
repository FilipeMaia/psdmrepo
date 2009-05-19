#ifndef O2OTRANSLATOR_O2OXTCITERATOR_H
#define O2OTRANSLATOR_O2OXTCITERATOR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OXtcIterator.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "pdsdata/xtc/XtcIterator.hh"
#include "pdsdata/xtc/TypeId.hh"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

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

namespace O2OTranslator {

class O2OXtcScannerI ;
class O2ODataTypeCvtI ;

class O2OXtcIterator : public Pds::XtcIterator {
public:

  // Default constructor
  O2OXtcIterator ( Xtc* xtc, O2OXtcScannerI* scanner ) ;

  // Destructor
  virtual ~O2OXtcIterator () ;

  // process one sub-XTC, returns >0 for success, 0 for error
  virtual int process(Xtc* xtc) ;

protected:

private:

  // Data members
  O2OXtcScannerI* m_scanner ;

  // Copy constructor and assignment are disabled by default
  O2OXtcIterator ( const O2OXtcIterator& ) ;
  O2OXtcIterator operator = ( const O2OXtcIterator& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OXTCITERATOR_H
