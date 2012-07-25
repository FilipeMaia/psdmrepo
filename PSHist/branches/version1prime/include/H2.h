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
 *  @brief Interface for 2-dimensional histogram class.
 * 
 *  Currently this interface defines only very simple filling operations.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see HManager
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class H2 : boost::noncopyable {
public:

  // Destructor
  virtual ~H2 () {}

  /**
   *  @brief Fill histogram.
   *  
   *  @param[in] x      Histogrammed value.
   *  @param[in] y      Histogrammed value.
   *  @param[in] weight Weight assigned to this value.
   */
  virtual void fill(double x, double y, double weight=1.0) = 0;

  /**
   *  @brief Fill histogram with multiple values.
   *  
   *  This method is equivalent to calling fill() multiple times but
   *  implementation may provide more efficient way if you have
   *  many values to fill histogram at once.
   *  
   *  @param[in] n      Size of the values array.
   *  @param[in] x      Histogrammed values array, array size is n.
   *  @param[in] y      Histogrammed values array, array size is n.
   *  @param[in] weight Weight assigned to these values, if pointer is 
   *                    zero then weight 1 is assumed for every value.
   */
  virtual void fillN(unsigned n, const double* x, const double* y, const double* weight=0) = 0;

  /**
   *  @brief Reset the accumulated contents of a histogram.
   */
  virtual void reset() = 0;

  /// Print some basic information about histogram to a stream
  virtual void print(std::ostream& o) const = 0;

};

} // namespace PSHist

#endif // PSHIST_H2_H
