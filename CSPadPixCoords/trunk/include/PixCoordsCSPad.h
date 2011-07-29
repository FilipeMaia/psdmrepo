#ifndef CSPADPIXCOORDS_PIXCOORDSCSPAD_H
#define CSPADPIXCOORDS_PIXCOORDSCSPAD_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsCSPad.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "CSPadPixCoords/PixCoordsQuad.h"
#include "PSCalib/CSPadCalibPars.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace CSPadPixCoords {

/**
 *
 *
 *
 *
 *
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

class PixCoordsCSPad  {
public:

  // Default constructor
  PixCoordsCSPad ( PixCoordsQuad *pix_coords_quad,  PSCalib::CSPadCalibPars *cspad_calibpar ) ;

  // Destructor
  virtual ~PixCoordsCSPad () ;

protected:

private:

  // Data members
  PixCoordsQuad           *m_pix_coords_quad;  
  PSCalib::CSPadCalibPars *m_cspad_calibpar;  

  // Copy constructor and assignment are disabled by default
  PixCoordsCSPad ( const PixCoordsCSPad& ) ;
  PixCoordsCSPad& operator = ( const PixCoordsCSPad& ) ;

};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_PIXCOORDSCSPAD_H
