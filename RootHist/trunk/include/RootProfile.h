#ifndef ROOTHIST_ROOTPROFILE_H
#define ROOTHIST_ROOTPROFILE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RootProfile.
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
#include "PSHist/Profile.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSHist/Axis.h"
#include "root/TProfile.h"

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
 *  @brief Implementation of PSHist::Profile interface.
 *  
 *  This class implements 1D profile-histogram with equal or variable bin size,
 *  which are defined in ROOT as TProfile.
 *  
 *  Essentially, constructor of this class creates an type-independent pointer TProfile *m_histp;
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

class RootProfile : public PSHist::Profile {
public:

  /**
   *  @brief Instantiate new profile histogram.
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] axis   X axis definition.
   *  @param[in] option Option string.
   */
  RootProfile ( const std::string& name, const std::string& title, 
      const PSHist::Axis& axis, const std::string& option="" );

  /**
   *  @brief Instantiate new profile histogram.
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] axis   X axis definition.
   *  @param[in] ylow   Lowest possible value for Y values.
   *  @param[in] yhigh  Highest possible value for Y values.
   *  @param[in] option Option string.
   */
  RootProfile ( const std::string& name, const std::string& title, 
      const PSHist::Axis& axis, double ylow, double yhigh, const std::string& option="" );

  // Destructor
  virtual ~RootProfile();

  /// Implementation of the corresponding method from PSHist::Profile interface.
  virtual void fill(double x, double y, double weight);

  /// Implementation of the corresponding method from PSHist::Profile interface.
  virtual void fillN(unsigned n, const double* x, const double* y, const double* weight);

  /// Implementation of the corresponding method from PSHist::Profile interface.
  virtual void reset();

  /// Implementation of the corresponding method from PSHist::Profile interface.
  virtual void print(std::ostream& o) const;

private:

  TProfile *m_histp;    ///< Corresponding ROOT Profile object

};

} // namespace RootHist

#endif // ROOTHIST_ROOTPROFILE_H
