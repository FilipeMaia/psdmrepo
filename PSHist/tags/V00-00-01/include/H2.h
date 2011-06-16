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

#include <string>
#include <iostream>

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
 *  PSHist is a fully abstract package for histogramming in PSANA
 *
 *  H2 is an abstract class which provides the final-package-implementation-independent
 *  interface to the 2D histograms. All methods of this class are virtual and should
 *  be implemented in derived package/class, i.e. RootHist/RootH2.
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

class H2 {
public:

  // Constructors


  // Default constructor
  H2 () { }

  // Constructors for equal and variable bin size histograms
  H2 ( int type, const std::string &name, const std::string &title, int nbinsx, double xlow, double xhigh, 
                                                                    int nbinsy, double ylow, double yhigh ) { }

  H2 ( int type, const std::string &name, const std::string &title, int nbinsx, double xlow, double xhigh, 
                                                                    int nbinsy, double *ybinedges ) { }

  H2 ( int type, const std::string &name, const std::string &title, int nbinsx, double *xbinedges,
                                                                    int nbinsy, double ylow, double yhigh ) { }

  H2 ( int type, const std::string &name, const std::string &title, int nbinsx, double *xbinedges,
                                                                    int nbinsy, double *ybinedges ) { }

  // Destructor
  virtual ~H2 () { std::cout << "H2::~H2 () : in destructor." << std::endl; }


  // Methids

  virtual void fill(double x, double y, double weight=1.0) = 0;

  virtual void reset() = 0;

  virtual void print(std::ostream &o) const = 0;

private:

  // Data members
  std::string m_title;
  
  // Copy constructor and assignment are disabled by default
  H2 ( const H2& ) ;
  H2& operator = ( const H2& ) ;

};

} // namespace PSHist

#endif // PSHIST_H2_H
