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

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSHist {

/**
 *  PSHist is a fully abstract package for histogramming in PSANA
 *
 *  H1 is an abstract class which provides the final-package-implementation-independent
 *  interface to the 1D histograms. All methods of this class are virtual and should
 *  be implemented in derived package/class, i.e. RootHist/RootH1.
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

class H1 {
public:

  // Constructors
  H1 () {
    //std::cout << "H1:: default constructor" << std::endl; 
  }

  // Destructor
  virtual ~H1 () { std::cout << "H1::~H1 () : in destructor." << std::endl; }


  // Methods

  virtual void fill(double x, double weight=1.0) = 0;

  virtual void reset() = 0;

  virtual void print(std::ostream &o) const = 0;

private:

  // Data members
  //std::string m_title;
  
  // Copy constructor and assignment are disabled by default
  H1 ( const H1& ) ;
  H1& operator = ( const H1& ) ;

};

} // namespace PSHist

#endif // PSHIST_H1_H
