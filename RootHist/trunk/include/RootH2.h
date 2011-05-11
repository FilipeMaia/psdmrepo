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

#include <iostream>
#include <string>

//----------------------
// Base Class Headers --
//----------------------

#include "PSHist/H2.h"
#include "PSHist/Axis.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "root/TH2.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace RootHist {

/**
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
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

template <typename HTYPE>
class RootH2 : public PSHist::H2 {
public:

  // Constructors
  RootH2<HTYPE> () {}


  RootH2<HTYPE> ( const std::string &name, const std::string &title, PSHist::Axis &xaxis, PSHist::Axis &yaxis );


  RootH2<HTYPE> ( const std::string &name, const std::string &title, int nbinsx, double xlow, double xhigh, 
	                                                             int nbinsy, double ylow, double yhigh );

  RootH2<HTYPE> ( const std::string &name, const std::string &title, int nbinsx, double xlow, double xhigh, 
	                                                             int nbinsy, double *ybinedges );

  RootH2<HTYPE> ( const std::string &name, const std::string &title, int nbinsx, double *xbinedges,
	                                                             int nbinsy, double ylow, double yhigh );

  RootH2<HTYPE> ( const std::string &name, const std::string &title, int nbinsx, double *xbinedges,
	                                                             int nbinsy, double *ybinedges );

  // Destructor
  virtual ~RootH2<HTYPE> () {}

  // Methods
  virtual void fill(double x, double y, double weight=1.0);
  virtual void reset();
  virtual void print(std::ostream &o) const;

private:

  // Data members
  TH2 *m_histp;

  // Static members
  //static int s_number_of_booked_histograms;

  // Copy constructor and assignment are disabled by default
  RootH2<HTYPE> ( const RootH2<HTYPE>& );
  RootH2<HTYPE>& operator = ( const RootH2<HTYPE>& );
};

} // namespace RootHist

#endif // ROOTHIST_ROOTH2_H
