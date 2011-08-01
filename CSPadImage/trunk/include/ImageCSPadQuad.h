#ifndef CSPADIMAGE_IMAGECSPADQUAD_H
#define CSPADIMAGE_IMAGECSPADQUAD_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImageCSPadQuad.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
//----------------------
// Base Class Headers --
//----------------------

#include "PSCalib/CSPadCalibPars.h"
#include "CSPadImage/QuadParameters.h"
#include "CSPadImage/Image2D.h"

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

template <typename T>
class ImageCSPadQuad  {
public:

  enum { NRows = 850 };
  enum { NCols = 850 };
  // enum { NRows = 1000 };
  // enum { NCols = 1000 };

  // Default constructor
  ImageCSPadQuad () ;

  // Regular constructor
  ImageCSPadQuad (const T* data, QuadParameters* quadpars, PSCalib::CSPadCalibPars *cspad_calibpar) ;

  // Destructor
  virtual ~ImageCSPadQuad () ;


  // Methods
  void fillQuadImage() ;
  void addSectionToQuadImage(uint32_t sect, float xcenter, float ycenter, float zcenter, float rotation) ;
  void saveQuadImageInFile() ;

  Image2D<T>* getQuadImage2D(){ return m_quad_image_2d; } ;

private:

  // Copy constructor and assignment are disabled by default
  ImageCSPadQuad ( const ImageCSPadQuad& ) ;
  ImageCSPadQuad operator = ( const ImageCSPadQuad& ) ;

//------------------
// Static Members --
//------------------

  // Data members
  const T        *m_data;
  QuadParameters *m_quadpars;
  T               m_quad_image[NRows][NCols];
  Image2D<T>     *m_quad_image_2d;
  PSCalib::CSPadCalibPars *m_cspad_calibpar;

  std::vector<int> v_image_shape;

  uint32_t m_n2x1;
  uint32_t m_nrows2x1;
  uint32_t m_ncols2x1;
  uint32_t m_sizeOf2x1Img;

  uint32_t m_roiMask;
};

} // namespace CSPadImage

#endif // CSPADIMAGE_IMAGECSPADQUAD_H
