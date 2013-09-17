#ifndef ROOTHISTOMANAGER_AXISDEF_H
#define ROOTHISTOMANAGER_AXISDEF_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AxisDef.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>

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

namespace RootHistoManager {

/**
 *  @ingroup RootHistoManager
 *  
 *  @brief Class defining a single axis (bins and limits) for histograms.
 *  
 *  Histogram axiz can be declared in a number of ways:
 *  - range (min and max) and number of bins 
 *  - range (min and max) and bin width
 *  - variable bins - lower edges of all bins
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Andrei Salnikov
 */

class AxisDef  {
public:

  /**
   *  @brief Create axis with fixed-size bins, given the axis range and
   *  number of bins.
   */ 
  AxisDef (double amin, double amax, int nbins) 
    : m_amin(amin), m_amax(amax), m_nbins(nbins) {}

  /**
   *  @brief Create axis with fixed-size bins, given the axis range and
   *  number of bins.
   */ 
  AxisDef (double amin, double amax, unsigned nbins)
    : m_amin(amin), m_amax(amax), m_nbins(nbins) {}

  /**
   *  @brief Create axis with fixed-size bins, given the axis range and
   *  single bin width.
   */ 
  AxisDef (double amin, double amax, double binWidth) 
    : m_amin(amin), m_amax(), m_nbins() 
  {
    m_nbins = int((amax-amin)/binWidth+0.5);
    m_amax = amin + m_nbins*binWidth;
  }

  /**
   *  @brief Create axis with fixed-size bins, given the axis range and
   *  single bin width.
   */ 
  AxisDef (double amin, double amax, float binWidth)
    : m_amin(amin), m_amax(), m_nbins() 
  {
    m_nbins = int((amax-amin)/binWidth+0.5);
    m_amax = amin + m_nbins*binWidth;
  }
  
  /**
   *  @brief Create variable bins with a vector specifying lower edges of the
   *  bins. 
   *  
   *  Number of bins will be equal to size of the vector minus one. 
   */ 
  AxisDef (const std::vector<double>& edges)
    : m_amin(0), m_amax(0), m_nbins(0), m_edges(edges) {}

  /// get range min
  double amin() const { return m_amin; }
  
  /// get range max
  double amax() const { return m_amax; }
  
  /// get number of bins
  unsigned nbins() const { return m_nbins; }
  
  /// get edges
  const std::vector<double>& edges() const { return m_edges; }
  
protected:

private:

  // Data members
  double m_amin;       ///< low edge of range, use only if m_edges is empty
  double m_amax;       ///< high edge of range, use only if m_edges is empty
  unsigned m_nbins;    ///< number of bins, use only if m_edges is empty
  std::vector<double> m_edges;  ///< if non-empty then defines edges of variable-width bins
};

} // namespace RootHistoManager

#endif // ROOTHISTOMANAGER_AXISDEF_H
