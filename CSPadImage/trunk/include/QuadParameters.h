#ifndef CSPADIMAGE_QUADPARAMETERS_H
#define CSPADIMAGE_QUADPARAMETERS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class QuadParameters.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
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

namespace CSPadImage {

/**
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

class QuadParameters  {
public:

  // Default constructor
  QuadParameters () ;

  // Regular constructor
  QuadParameters (uint32_t         quadNumber, 
                  std::vector<int> image_shape, 
                  size_t           nrows=850, 
                  size_t           ncols=850, 
                  uint32_t         numAsicsStored=16, 
                  uint32_t         roiMask=255) ;

  // Destructor
  virtual ~QuadParameters () ;

  size_t           getNRows             (){ return m_nrows;   };
  size_t           getNCols             (){ return m_ncols;   };
  uint32_t         getRoiMask           (){ return m_roiMask; };
  uint32_t         getQuadNumber        (){ return m_quadNumber; };
  uint32_t         getNumberAsicsStroed (){ return m_numAsicsStored; };
  std::vector<int> getImageShapeVector  (){ return v_image_shape; };

  void print ();

private:

  // Copy constructor and assignment are disabled by default
  QuadParameters ( const QuadParameters& ) ;
  QuadParameters operator = ( const QuadParameters& ) ;

//------------------
// Static Members --
//------------------

  // Data members
  uint32_t         m_quadNumber;
  std::vector<int> v_image_shape;
  size_t           m_nrows;
  size_t           m_ncols;
  uint32_t         m_numAsicsStored;
  uint32_t         m_roiMask;

};

} // namespace CSPadImage

#endif // CSPADIMAGE_QUADPARAMETERS_H
