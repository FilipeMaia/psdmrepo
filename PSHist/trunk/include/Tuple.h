#ifndef PSHIST_TUPLE_H
#define PSHIST_TUPLE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Tuple.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSHist {

/**
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class Tuple  {
public:

  // Default constructor
  Tuple () ;

  // Destructor
  virtual ~Tuple () ;

  // Selectors (const)

  // Modifiers


private:

  // Copy constructor and assignment are disabled by default
  Tuple ( const Tuple& ) ;
  Tuple& operator = ( const Tuple& ) ;

  // Data members
  
  // Static Members
};

} // namespace PSHist

#endif // PSHIST_TUPLE_H
