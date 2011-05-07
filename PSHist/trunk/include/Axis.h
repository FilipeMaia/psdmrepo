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

#include <iostream> // for std::cout and std::endl

//#include <vector>
//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSHist {

/**
 *  Axis class defines the binning parameters for H1 and H2 histogram axes.
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

class Axis {
public:

  // Constructors

  /**
   *  Create axis with fixed bin sizes, given the axis range and number of bins.
   */ 
  Axis (int nbins, double amin, double amax) 
    : m_amin(amin), m_amax(amax), m_nbins(nbins), m_edges(0) {}

  /**
   *  Create axis with variable bin sizes, defined by the input array of bin edges. 
   *  Number of the bin edges is equal to the number of bins + 1.
   */ 
  Axis (int nbins, const double *edges)
    : m_amin(0), m_amax(0), m_nbins(nbins), m_edges(edges) {}

  // Destructor
  virtual ~Axis () { std::cout << "Axis::~Axis () : in destructor." << std::endl; }

  // Methods

  // get range min
  double amin() const { return m_amin; }
  
  // get range max
  double amax() const { return m_amax; }
  
  // get number of bins
  int nbins() const { return m_nbins; }
  
  // get edges
  const double* edges() { return m_edges; }

  // print data members
  void print() const 
  {
    std::cout << "=========================================================" << std::endl;
    if ( m_edges == 0 ) std::cout << "Axis with equal bin sizes"    << std::endl;
    else                std::cout << "Axis with variable bin sizes" << std::endl;
    std::cout << "Axis::m_nbins=" << m_nbins
              << "      m_amin="  << m_amin
              << "      m_amax="  << m_amax  << std::endl;     
    if ( m_edges != 0 ) std::cout << "Axis::m_edges[0]=" << m_edges[0]        
                                  << "      m_edges[N]=" << m_edges[m_nbins] << std::endl;     
    std::cout << "=========================================================" << std::endl;
  }

private:

  // Data members
  double        m_amin;
  double        m_amax;
  int           m_nbins;
  const double *m_edges;
  
  // Copy constructor and assignment are disabled by default
  Axis ( const Axis& ) ;
  Axis& operator = ( const Axis& ) ;

};

} // namespace PSHist

#endif // PSHIST_AXIS_H
