#ifndef PSHIST_AXIS_H
#define PSHIST_AXIS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Axis.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <iosfwd>

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSHist {

/**
 *  @ingroup PSHist
 * 
 *  @brief Axis class defines the binning parameters for H1 and H2 histogram axes.
 *  
 *  This class can be used to define both same-width and variable-width
 *  binnings for histograms. Type of the binning is defined by the constructor
 *  used to instantiate Axis object.
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

class Axis {
public:

  // Constructors

  /**
   *  @brief Create axis with fixed bin sizes
   *  
   *  @param[in] nbins  Number of bins.
   *  @param[in] amin   Low edge of the first bin.
   *  @param[in] amax   High edge of the last bin.
   *  
   *  @throw ExceptionBins thrown when number of bins is 0
   *  @throw ExceptionAxisRange thrown when amin is equal or higher that amax
   */ 
  Axis (unsigned nbins, double amin, double amax);

  /**
   *  @brief Create axis with variable bin sizes
   *
   *  @param[in] nbins  Number of bins.
   *  @param[in] edges  Array of the histogram edges, size of the array 
   *                    is @c nbins+1, it should contain ordered values for
   *                    low edges of all bins plus high edge of last bin. 
   *  
   *  @throw ExceptionBins thrown when number of bins is 0
   *  @throw ExceptionAxisEdgeOrder thrown when edges are not ordered
   */ 
  Axis (unsigned nbins, const double *edges);

  /// Get low edge of range, makes sense to call only for fixed-width bins.
  double amin() const { return m_amin; }
  
  /// Get high edge of range, makes sense to call only for fixed-width bins.
  double amax() const { return m_amax; }
  
  /// Get number of bins.
  unsigned nbins() const { return m_nbins; }
  
  /// Get array of edges, will return zero pointer for fixed-width bins.
  const double* edges() const;

  // print data members
  void print(std::ostream& out) const;
  
private:

  // Data members
  double        m_amin;
  double        m_amax;
  unsigned      m_nbins;
  std::vector<double> m_edges;
  
};

} // namespace PSHist

#endif // PSHIST_AXIS_H
