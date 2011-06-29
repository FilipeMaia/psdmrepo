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
#include <iosfwd>
#include <boost/utility.hpp>

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
 *  @ingroup PSHist
 *  
 *  @brief Interface for n-tuple class.
 * 
 *  Currently this interface defines only very simple filling operations.
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

class Tuple : boost::noncopyable {
public:

  // Destructor
  virtual ~Tuple () {}

  /**
   *  @brief Add column to tuple.
   *  
   *  @throw ExceptionDuplicateColumn thrown if column name already defined
   */
  virtual Column* column( const std::string& name, void* address, const std::string& columnlist ) = 0;

  /**
   *  @brief Add column to tuple.
   */
  virtual Column* column( void* address, const std::string& columnlist ) = 0; // for auto-generated name

  virtual void fill() = 0;

  virtual void reset() = 0;

  virtual void print(std::ostream& o) const = 0;

};

} // namespace PSHist

#endif // PSHIST_TUPLE_H
