#ifndef PSHIST_COLUMN_H
#define PSHIST_COLUMN_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Column.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include <string>
#include <iostream>

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
 *  PSHist is a fully abstract package for histogramming in PSANA
 *  Column is an abstract class which provides the final-package-implementation-independent
 *  interface to the N-tuple-like parameter. All methods of this class are virtual and should
 *  be implemented in derived package/class, i.e. RootHist/RootColumn.
 *   
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class Column  {
public:

  // Default constructor
  Column () {}

  // Destructor
  virtual ~Column () {}

  // Selectors (const)

  // Modifiers

  virtual void print(std::ostream &o) const = 0;


private:

  // Copy constructor and assignment are disabled by default
  Column ( const Column& ) ;
  Column& operator = ( const Column& ) ;

  // Data members
  
  // Static Members
};

} // namespace PSHist

#endif // PSHIST_COLUMN_H
