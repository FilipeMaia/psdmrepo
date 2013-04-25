#ifndef O2OTRANSLATOR_O2OXTCVALIDATOR_H
#define O2OTRANSLATOR_O2OXTCVALIDATOR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OXtcValidator.
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

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *  XTC iterator which validates XTC structure 
 *
 *  This software was developed for the LUSI project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class O2OXtcValidator : public Pds::XtcIterator {
public:

  // Default constructor
  O2OXtcValidator () ;

  // Destructor
  virtual ~O2OXtcValidator () ;

  // process one sub-XTC, returns >0 for success, 0 for error
  virtual int process(Xtc* xtc) ;

protected:

private:

  int m_status ;

  // Copy constructor and assignment are disabled by default
  O2OXtcValidator ( const O2OXtcValidator& ) ;
  O2OXtcValidator& operator = ( const O2OXtcValidator& ) ;
  
};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OXTCVALIDATOR_H
