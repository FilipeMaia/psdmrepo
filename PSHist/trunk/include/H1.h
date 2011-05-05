#ifndef PSHIST_H1_H
#define PSHIST_H1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class H1.
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
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class H1 {
public:

  // Constructors
  H1 () { 
    //std::cout << "H1:: default constructor" << std::endl; 
  }

//H1 ( std::string &title ) : m_title(title) {}

  H1 ( int type, const std::string &name, const std::string &title, int nbins, double xlow, double xhigh ) { 
    //std::cout << "H1:: default constractor for equidistant bins" << std::endl;
  }

  H1 ( int type, const std::string &name, const std::string &title, int nbins, double *xbinedges ) {
    //std::cout << "H1:: default constractor for variable bin size" << std::endl;
  }

  // Destructor
  virtual ~H1 () { std::cout << "H1::~H1 () : in destructor." << std::endl; }

  virtual void fill(double x, double weight=1.0) = 0;

  virtual void reset() = 0;

  virtual void print(std::ostream &o) const = 0;

private:

  // Data members
  std::string m_title;
  
  // Copy constructor and assignment are disabled by default
  H1 ( const H1& ) ;
  H1& operator = ( const H1& ) ;

};

} // namespace PSHist

#endif // PSHIST_H1_H
