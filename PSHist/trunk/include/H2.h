#ifndef PSHIST_H2_H
#define PSHIST_H2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class H2.
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

class H2  {
public:

  // Default constructor
  H2 () ;

  // Destructor
  virtual ~H2 () ;

protected:

private:

  // Data members
  
  int m_memberVariable;  // private members start with m_

  // Copy constructor and assignment are disabled by default
  H2 ( const H2& ) ;
  H2& operator = ( const H2& ) ;

//------------------
// Static Members --
//------------------

public:

  // Selectors (const)

  // Modifiers

private:

  // Data members
  static int s_staticVariable;     // Static data member starts with s_.

};

} // namespace PSHist

#endif // PSHIST_H2_H
