#ifndef ROOTHIST_ROOTH1_H
#define ROOTHIST_ROOTH1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RootH1.
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
#include "PSHist/H1.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSHist/Axis.h"
#include "root/TH1.h"

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
 *  @brief Implementation of PSHist::H1 interface.
 *  
 *  This class implements 1D histograms with equal or variable bin size,
 *  which are defined in ROOT as TH1I, TH1F, and TH1D.
 *  
 *  Essentially, constructor of this class creates an type-independent pointer TH1 *m_histp;
 *  which is used for any other interactions with ROOT histograms of differnt types.
 *  
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

template <typename HTYPE>
class RootH1 : public PSHist::H1 {
public:

  /**
   *  @brief Instantiate new histogram.
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] axis   Axis definition.
   */
  RootH1<HTYPE>(const std::string& name, const std::string& title, const PSHist::Axis& axis);

  // Destructor
  virtual ~RootH1<HTYPE>();

  /// Implementation of the corresponding method from PSHist::H1 interface.
  virtual void fill(double x, double weight);
  
  /// Implementation of the corresponding method from PSHist::H1 interface.
  virtual void fillN(unsigned n, const double* x, const double* weight);

  /// Implementation of the corresponding method from PSHist::H1 interface.
  virtual void reset();

  /// Implementation of the corresponding method from PSHist::H1 interface.
  virtual void print(std::ostream& o) const;

private:

  TH1 *m_histp;   ///< Corresponding ROOT histogram object

};

} // namespace RootHist

#endif // ROOTHIST_ROOTH1_H
