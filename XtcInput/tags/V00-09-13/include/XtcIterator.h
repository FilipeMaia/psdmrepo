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
#include <stack>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/xtc/Xtc.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief XTC iterator which does recursive iteration and returns every 
 *  individual XTC object.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class XtcIterator {
public:

  // Constructor takes pointer to Xtc object
  XtcIterator ( Pds::Xtc* xtc ) ;

  // Destructor
  ~XtcIterator () ;

  // Returns next XTC or 0
  Pds::Xtc* next() ;

protected:

private:

  // Data members
  Pds::Xtc* m_initial;
  std::stack<Pds::Xtc*> m_xtc;
  std::stack<int> m_off;

  // Copy constructor and assignment are disabled by default
  XtcIterator ( const XtcIterator& ) ;
  XtcIterator operator = ( const XtcIterator& ) ;

};

} // namespace XtcInput

#endif // XTCINPUT_XTCITERATOR_H
