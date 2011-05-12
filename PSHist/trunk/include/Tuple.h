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

#include <string>
#include <iostream>

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "PSHist/Column.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSHist {

/**
 *  PSHist is a fully abstract package for histogramming in PSANA
 *  Tuple is an abstract class which provides the final-package-implementation-independent
 *  interface to the N-tuple-like object. All methods of this class are virtual and should
 *  be implemented in derived package/class, i.e. RootHist/RootTuple.
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

class Tuple  {
public:

  // Default constructor
  Tuple () {}

  // Destructor
  virtual ~Tuple () {}

  // Selectors (const)

  // Modifiers

  virtual Column* column( const std::string &name, void* address, const std::string &columnlist ) = 0;

  virtual Column* column( void* address, const std::string &columnlist ) = 0; // for auto-generated name

  virtual void fill() = 0;

  virtual void reset() = 0;

  virtual void print(std::ostream &o) const = 0;


private:

  // Copy constructor and assignment are disabled by default
  Tuple ( const Tuple& ) ;
  Tuple& operator = ( const Tuple& ) ;

  // Data members
  
  // Static Members
};

} // namespace PSHist

#endif // PSHIST_TUPLE_H
