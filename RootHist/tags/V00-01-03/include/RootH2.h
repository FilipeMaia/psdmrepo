#ifndef ROOTHIST_ROOTH2_H
#define ROOTHIST_ROOTH2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RootH2.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <iosfwd>

//----------------------
// Base Class Headers --
//----------------------
#include "PSHist/H2.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSHist/Axis.h"
#include "root/TH2.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace RootHist {

/**
 *  @ingroup RootHist
 *  
 *  @brief Implementation of PSHist::H2 interface.
 *  
 *  This class implements 2D histograms with equal or variable bin size,
 *  which are defined in ROOT as TH2I, TH2F, and TH2D.
 *  
 *  Essentially, constructor of this class creates an type-independent pointer TH2 *m_histp;
 *  which is used for any other interactions with ROOT histograms of differnt types.
 *  
 *  
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

template <typename HTYPE>
class RootH2 : public PSHist::H2 {
public:

  /**
   *  @brief Instantiate new histogram.
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] xaxis  X axis definition.
   *  @param[in] yaxis  Y axis definition.
   */
  RootH2<HTYPE> ( const std::string& name, const std::string& title,
                  const PSHist::Axis& xaxis, const PSHist::Axis& yaxis );

  // Destructor
  virtual ~RootH2<HTYPE> ();

  /// Implementation of the corresponding method from PSHist::H2 interface.
  virtual void fill(double x, double y, double weight);
  
  /// Implementation of the corresponding method from PSHist::H2 interface.
  virtual void fillN(unsigned n, const double* x, const double* y, const double* weight);

  /// Implementation of the corresponding method from PSHist::H2 interface.
  virtual void reset();

  /// Implementation of the corresponding method from PSHist::H2 interface.
  virtual void print(std::ostream& o) const;

private:

  // Data members
  TH2 *m_histp;

};

} // namespace RootHist

#endif // ROOTHIST_ROOTH2_H
