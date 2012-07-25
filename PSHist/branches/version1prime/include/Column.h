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
#include <iosfwd>
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

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSHist {

/**
 *  @ingroup PSHist
 * 
 *  @brief Interface for tuple column class.

 *  Column is an abstract class which provides the final-package-implementation-independent
 *  interface to the N-tuple-like parameter. All methods of this class are virtual and should
 *  be implemented in derived package/class, i.e. RootHist/RootColumn.
 *   
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see Tuple
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class Column : boost::noncopyable {
public:

  // Destructor
  virtual ~Column () {}

  /// Print some basic information about column to a stream
  virtual void print(std::ostream& o) const = 0;

};

} // namespace PSHist

#endif // PSHIST_COLUMN_H
