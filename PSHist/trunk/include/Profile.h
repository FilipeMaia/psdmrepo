#ifndef PSHIST_PROFILE_H
#define PSHIST_PROFILE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Profile.
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
 *  Profile is an abstract class which provides the final-package-implementation-independent
 *  interface to the 1D profile-histogram. All methods of this class are virtual and should
 *  be implemented in derived package/class, i.e. RootHist/RootProfile.
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

class Profile {
public:

  // Constructors
  Profile () {}

  // Destructor
  virtual ~Profile () {}


  // Methods
  virtual void fill(double x, double y, double weight=1.0) = 0;

  virtual void reset() = 0;

  virtual void print(std::ostream &o) const = 0;

private:

  // Data members
  
  // Copy constructor and assignment are disabled by default
  Profile ( const Profile& ) ;
  Profile& operator = ( const Profile& ) ;

};

} // namespace PSHist

#endif // PSHIST_PROFILE_H
